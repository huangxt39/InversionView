import re
import os
import sys
import uuid
from copy import deepcopy
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.utils import logging
from transformer_lens import HookedTransformer
from datasets import DatasetDict

logger = logging.get_logger(__name__)


def get_paths(args):
    if args.save_dir == "random":
        exp_save_dir = f"./data_and_model/{str(uuid.uuid4())}"
    else:
        exp_save_dir = f"./data_and_model/{args.save_dir}"

    if not os.path.exists(exp_save_dir):
        os.mkdir(exp_save_dir)

    if args.probed_task == "ioi":
        max_len = 40
        probed_model_path = "gpt2"
        data_path = "../datasets/ioi_prompts.pkl"    # first run data generation code
    elif args.probed_task == "addition":
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from addition.train import MAX_LEN
        max_len = MAX_LEN - 1   # minus eos
        probed_model_path = "../training_outputs/addition_fixed/checkpoint-59350"
        data_path = None
    elif args.probed_task == "counting":
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from counting.train import MAX_LEN
        max_len = MAX_LEN - 1   # minus eos
        probed_model_path = "../training_outputs/counting/checkpoint-914100"
        data_path = None
    elif args.probed_task == "fact":
        max_len = 40   
        probed_model_path = "gpt2-xl"
        data_path = "../datasets/factual/data_all"
    
        

    return probed_model_path, data_path, exp_save_dir, max_len



class rolloutManagerIOI:
    def __init__(self, init_path):
        with open(init_path, "rb") as f:
            rollouts: list[dict[str, str]] = pickle.load(f)
        # {'[PLACE]': 'hospital', '[OBJECT]': 'basketball', 'text': 'The local big hospital Jacob and Jason went to had a basketball. Jason gave it to Jacob', 'IO': 'Jacob', 'S': 'Jason', 'TEMPLATE_IDX': 11, 'pos_IO': 4, 'pos_IO-1': 3, 'pos_IO+1': 5, 'pos_S': 6, 'pos_S-1': 5, 'pos_S+1': 7, 'pos_S2': 13, 'pos_end': 16, 'pos_starts': 0, 'pos_punct': 12}
        def check_if_ioi(obj):
            tokens = obj["text"].split(" ")
            return (tokens.count(obj["IO"]) == 2) and (tokens.count(obj["S"]) == 2)
        
        self.rollouts = list(filter(check_if_ioi, rollouts))
        self.cursor = 0
    
    def next_batch(self, bz):
        result = list(map(lambda x: x["text"], self.rollouts[self.cursor: self.cursor+bz]))
        self.cursor += bz
        if len(result) < bz:
            num_lacked = bz - len(result)
            result.extend( list(map(lambda x: x["text"], self.rollouts[:num_lacked])) )
            self.cursor = num_lacked
            print("warning: reuse data")
        return result


class rolloutManagerAddition:
    def __init__(self, dataset):
        self.data = dataset.data
        random.shuffle(self.data)
        self.cursor = 0

    def next_batch(self, bz):

        def convert_to_text(pairs):
            return [str(a) + "+" + str(b) + "=" + str(a+b) for a,b in pairs]
        
        result = self.data[self.cursor: self.cursor+bz]
        self.cursor += bz
        if len(result) < bz:
            num_lacked = bz - len(result)
            result.extend( self.data[:num_lacked] )
            self.cursor = num_lacked
            print("warning: reuse data")
        
        result = convert_to_text(result)
        return result
    

class rolloutManagerCounting:
    def __init__(self, dataset):
        self.data = dataset.data    # i, j, sum, label
        random.shuffle(self.data)
        self.cursor = 0

    def next_batch(self, bz):
        
        result = self.data[self.cursor: self.cursor+bz]
        self.cursor += bz
        if len(result) < bz:
            num_lacked = bz - len(result)
            result.extend( self.data[:num_lacked] )
            self.cursor = num_lacked
            print("warning: reuse data")
        
        return result



class rolloutManagerFact:
    def __init__(self, dataset: DatasetDict):
        self.dataset = {k: v["sentence"] for k, v in dataset.items()}
        self.cursor = {k: 0 for k in dataset}
    
    def next_batch(self, bz: int, probed_act: str, last_pos: bool):
        block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
        head_idx = int(re.search(r"hook_result\.(\d+)", probed_act).group(1))
        h_key = f"{block_idx}.{head_idx}" if last_pos else f"{block_idx}.{head_idx}_all"
        
        if self.cursor[h_key]+bz > len(self.dataset[h_key]):
            result = self.dataset[h_key][self.cursor[h_key]: len(self.dataset[h_key])].copy()
            
            num_lacked = bz - len(result)
            random.shuffle(self.dataset[h_key])
            result.extend( self.dataset[h_key][:num_lacked] )
            self.cursor[h_key] = num_lacked
        else:
            result = self.dataset[h_key][self.cursor[h_key]: self.cursor[h_key]+bz]
            self.cursor[h_key] += bz
        
        return result

def retrieve_act(probed_act, cache):
    if "hook_result" not in probed_act:
        activation = cache[probed_act]
    else:
        probed_act, head_idx = probed_act.rsplit(".", maxsplit=1)
        head_idx = int(head_idx)
        activation = cache[probed_act][:, :, head_idx].clone()


    return activation

def add_necessary_hooks(hooked_model: HookedTransformer, probed_acts: list[str]):
    hook_name = []
    for probed_act in probed_acts:
        if "hook_result" in probed_act:
            probed_act, _ = probed_act.rsplit(".", maxsplit=1)
            # for i in range(hooked_model.cfg.n_layers):
            block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
            hooked_model.blocks[block_idx].attn.cfg.use_attn_result = True

            hook_name.append(f"blocks.{block_idx}.attn.hook_pattern")
        hook_name.append(probed_act)
    hook_name = set(hook_name)

    hooked_model.reset_hooks()
    cache = hooked_model.add_caching_hooks(lambda n: n in hook_name)
    return cache

def end_caching(hooked_model: HookedTransformer):
    for i in range(hooked_model.cfg.n_layers):  # cache attn result costs memory, turn off when it's unnecessary
        hooked_model.blocks[i].attn.cfg.use_attn_result = False
    hooked_model.reset_hooks()  # clean up hooks

def split_groups(probed_acts: list[str]):
    groups = []
    probed_acts = deepcopy(probed_acts)
    if "blocks.0.hook_resid_pre" in probed_acts:
        probed_acts.remove("blocks.0.hook_resid_pre")
        groups.append(["blocks.0.hook_resid_pre"])
    
    block_ids = [int(re.search(r"blocks\.(\d+)\.", a).group(1)) for a in probed_acts]
    groups_temp = {i:[] for i in set(block_ids)}
    for block_idx, probed_act in zip(block_ids, probed_acts):
        groups_temp[block_idx].append(probed_act)
    groups.extend(list(groups_temp.values()))

    return groups

class ActivationDataset(Dataset):
    def __init__(self, data_per_epoch: int, probed_acts: list):
        super().__init__()

        self.probed_acts = probed_acts
        self.data_per_epoch = data_per_epoch 
        self.re_init()

    def re_init(self):
        self.activations = None
        self.tokens = None
        self.act_site_ids = None
        self.initialized = False
        self.cursor = 0
        self.instance_per_act_site = {probed_act: 0 for probed_act in self.probed_acts}

    
    def add_data(self, act: torch.FloatTensor, tkn: list[list[int]], name: str):
        if not self.initialized:
            self.activations = torch.zeros(self.data_per_epoch, act.size(1), dtype=act.dtype, device=act.device)
            self.tokens = []
            self.act_site_ids = torch.empty(self.data_per_epoch, dtype=torch.long, device=act.device)
            self.initialized = True
        
        batch_len = act.size(0)
        num_discard = max(0, batch_len + self.cursor - self.data_per_epoch)
        if num_discard == batch_len:
            return 0
        elif num_discard > 0:
            act = act[:-num_discard]
            tkn = tkn[:-num_discard]
            batch_len = len(act)
        assert act.size(0) == len(tkn)
        self.activations[self.cursor: self.cursor+batch_len] = act
        self.tokens.extend(tkn)
        self.act_site_ids[self.cursor: self.cursor+batch_len] = self.probed_acts.index(name)
        self.instance_per_act_site[name] += batch_len
        self.cursor += batch_len
        assert self.cursor <= self.activations.size(0)
        return batch_len

    def __getitem__(self, index):
        assert self.initialized
        return self.activations[index], self.tokens[index], self.act_site_ids[index]
    
    def __len__(self):
        return self.cursor
    

class customCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features, return_tensors=None):
        return self.torch_call(features)

    def torch_call(self, examples):
        # examples = sorted(examples, key=lambda x: x[-1])
        act, tkn, s_id = tuple(zip(*examples))
        act, s_id = torch.vstack(act), torch.hstack(s_id)

        max_len = max(map(lambda x: len(x), tkn))
        for s in tkn:
            s.extend([self.tokenizer.pad_token_id] * (max_len-len(s)))
        padded_tkn = torch.LongTensor(tkn)

        batch = {"input_ids": padded_tkn, "activation": act, "act_site_ids": s_id, "use_cache": False}

        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
