from transformers import AutoTokenizer, PreTrainedTokenizerFast, TrainerCallback, TrainingArguments
import math
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import os
import pickle
import re
import uuid
from typing import Optional
from contextlib import redirect_stdout
from glob import glob
from copy import deepcopy
import string
import json
from collections import OrderedDict

from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainerCallback, TrainerControl, TrainerState, GenerationConfig
from transformers.generation.utils import GenerationMode
from transformer_lens import HookedTransformer

from utils import *
from model import CustomGPT2LMHeadModel

# logging.basicConfig(level=logging.INFO)



def string_with_marked_position(tokenizer: PreTrainedTokenizerFast, token_ids: list[int], selected_idx: int):
    if len(tokenizer.vocab) == 50259: # gpt2
        str_tokens = list(map(lambda x: x.replace("Ġ", " "),  tokenizer.convert_ids_to_tokens(token_ids)))
        delimiter = ""
    else:
        str_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        delimiter = " "
    
    if selected_idx >= 0 and selected_idx < len(str_tokens):
        str_tokens[selected_idx] = "(" + str_tokens[selected_idx] + ")"

    s = delimiter.join(str_tokens)

    return s

def get_test_activation(args, probed_act: str, hooked_model: HookedTransformer, text: Optional[str] = None, sel_idx: Optional[list[int]] = None):
    # single rollout, multiple positions
    if args.probed_task == "ioi":
        default_text = "Then, Mary and John went to the supermarket. John gave a drink to Mary"
        default_sel_idx = [-2]
    elif args.probed_task == "addition":
        default_text = "615+861=1476"
        default_sel_idx = [-5]
    elif args.probed_task == "counting":
        default_text = "vvzccvczvvvzvcvc|v:8"
        default_sel_idx = [-2]
    elif args.probed_task == "fact":
        default_text = "Beats Music is owned by Apple"
        default_sel_idx = [3, -2]


    if text is None:
        text = default_text
    if sel_idx is None:
        sel_idx = default_sel_idx

    cache = add_necessary_hooks(hooked_model, [probed_act])
    tokens = hooked_model.to_tokens(text)
    with torch.no_grad():
        hooked_model(tokens, return_type=None)

    act = retrieve_act(probed_act, cache).squeeze(0)

    sel_act = []
    for i in sel_idx:
        if i < 0:
            i += tokens.size(1)
        sel_act.append((act[i], string_with_marked_position(hooked_model.tokenizer, tokens[0].tolist(), i), probed_act, act[0]))
    
    end_caching(hooked_model)
    return sel_act

@torch.no_grad()
def generate_and_print(
        args,
        test_act: list[tuple[torch.FloatTensor, str, str, torch.FloatTensor]], 
        model: CustomGPT2LMHeadModel, 
        tokenizer: PreTrainedTokenizerFast, 
        gen_mode: Optional[str] = "greedy",
    ):

    for i, (act, info, probed_act, no_info_act) in enumerate(test_act):
        print("\n", "="*50)
        
        act = act.unsqueeze(0).expand(args.gen_num, -1).to(model.device)
        no_info_act = no_info_act.unsqueeze(0).expand(args.gen_num, -1).to(model.device)
        act_site_id = torch.full((args.gen_num,), model.probed_acts.index(probed_act), device=model.device)
           
        if gen_mode == "sample":
            temperature = args.temperature
        else:
            temperature = None
        generated_tokens = model.generate(act, act_site_id, gen_mode, temperature=temperature)
        
        print(f"\nexample {i}:\n\t", info)
        for j in range(generated_tokens.size(0)):
            filtered_t = generated_tokens[j].tolist()
            filtered_t = list(filter(lambda x: x!=tokenizer.pad_token_id and x!=tokenizer.eos_token_id, filtered_t))
            filtered_s = string_with_marked_position(tokenizer, filtered_t, -1)
            print("\nsorted:\n\t", filtered_s, "\n")


@torch.no_grad()
def run_test(
        args,
        probed_acts: list[str], 
        test_loader: dict[str, DataLoader], 
        model: CustomGPT2LMHeadModel, 
        tokenizer: PreTrainedTokenizerFast, 
        hooked_model: HookedTransformer, 
        max_len: Optional[int] = None,
    ):
    device = model.device
    gen_mode = "sample"
    print("============ testing generation quality ==============")
    all_ratio_good = OrderedDict()
    for probed_act in tqdm(probed_acts):
        cache = add_necessary_hooks(hooked_model, [probed_act])
        block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))

        num_good_gen = 0
        for input_batch in test_loader[probed_act]:
            activation = input_batch["activation"].to(device)
            act_site_ids = input_batch["act_site_ids"].to(device)
            assert (act_site_ids == model.probed_acts.index(probed_act)).all()

            generated_tokens = model.generate(activation, act_site_ids, gen_mode, max_len)
            
            hooked_model(generated_tokens, stop_at_layer=block_idx+1)

            recomputed_act = retrieve_act(probed_act, cache)

            m = (generated_tokens==tokenizer.pad_token_id) | (generated_tokens==tokenizer.eos_token_id)
            if args.cossim:
                sim = F.cosine_similarity(activation.unsqueeze(1), recomputed_act, dim=-1)
                sim.masked_fill_(m, -1.0)
                best_values, _ = sim.max(dim=1)
                num_good_gen += (best_values > 0.9).sum().item()
            else:
                diff = torch.linalg.vector_norm(activation.unsqueeze(1) - recomputed_act, dim=-1) / torch.linalg.vector_norm(activation.unsqueeze(1), dim=-1)
                diff.masked_fill_(m, 1e6)
                best_values, _ = diff.min(dim=1)
                num_good_gen += (best_values < 0.1).sum().item()
        
        ratio_good = num_good_gen / len(test_loader[probed_act].dataset)
        all_ratio_good[probed_act] = ratio_good

        end_caching(hooked_model)

    ratios = torch.tensor(list(all_ratio_good.values()))
    prob_weight = (ratios.mean() - ratios) * args.rebalance
    prob_weight = OrderedDict({k: v for k, v in zip(all_ratio_good.keys(), prob_weight.tolist())})

    for probed_act in probed_acts:
        print(probed_act.ljust(30), f"  ratio: {all_ratio_good[probed_act]:.3f} \t weight: {math.exp(prob_weight[probed_act]):.3f}")
    avg = torch.tensor(list(all_ratio_good.values())).mean().item()
    print(f"average \t {avg:.3f} \n")

    return prob_weight

        
                
