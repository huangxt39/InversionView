from transformers import AutoTokenizer, PreTrainedTokenizerFast, TrainerCallback, TrainingArguments, GPT2Tokenizer
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

from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer

from utils import *
from model import CustomGPT2LMHeadModel

# logging.basicConfig(level=logging.INFO)

byte_decoder = GPT2Tokenizer.from_pretrained("gpt2").byte_decoder

def rearrange_for_nonascii(tokens):
    tokens = tokens.copy()
    for i in range((len(tokens))):
        decoded = bytearray([byte_decoder[c] for c in tokens[i]]).decode("utf-8", errors="replace") 
        if "�" in decoded and (i < len(tokens)-1) and len(tokens[i]) < 4:
            t = tokens[i].lstrip("Ġ")
            tokens[i] = "Ġ_" if tokens[i].startswith("Ġ") else "_"
            tokens[i+1] = t + tokens[i+1]
        
    return tokens

def convert_to_str_tokens(tokenizer: PreTrainedTokenizerFast, token_ids: list[int]):
    if len(tokenizer.vocab) == 50259: # gpt2
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        tokens = rearrange_for_nonascii(tokens)
        str_tokens = [bytearray([byte_decoder[c] for c in t]).decode("utf-8", errors="replace") for t in tokens]
    else:
        str_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return str_tokens

def get_test_activation(probed_acts: list[str], hooked_model: HookedTransformer, text: str, sel_idx: list[int]):

    cache = add_necessary_hooks(hooked_model, probed_acts + [f"blocks.{i}.attn.hook_pattern" for i in range(hooked_model.cfg.n_layers)])
    tokens = hooked_model.to_tokens(text)
    with torch.no_grad():
        hooked_model(tokens, return_type=None)

    sel_act = []
    for probed_act in probed_acts:
        act = retrieve_act(probed_act, cache).squeeze(0)

        for i in sel_idx:
            if i < 0:
                i += tokens.size(1)

            if "hook_result" in probed_act:
                block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
                head_idx = int(re.search(r"hook_result\.(\d+)", probed_act).group(1))
                attn_weights = cache[f"blocks.{block_idx}.attn.hook_pattern"][0, head_idx, i].tolist()
                attn_weights = [round(w, 3) for w in attn_weights]
            else:
                attn_weights = []

            sel_act.append((act[i], probed_act, act[0], convert_to_str_tokens(hooked_model.tokenizer, tokens[0].tolist()), i, attn_weights))

    end_caching(hooked_model)
    return sel_act

@torch.no_grad()
def generate_and_print(
        args,
        test_act: list[tuple[torch.FloatTensor, str, str, torch.FloatTensor]], 
        model: CustomGPT2LMHeadModel, 
        tokenizer: PreTrainedTokenizerFast, 
        hooked_model: HookedTransformer,
    ):
    gen_mode = "sample"
    print("generating...")
    for i, (act, probed_act, no_info_act, q_input, q_pos, attn_w) in enumerate(tqdm(test_act)):
        gen_num = 100
        act = act.unsqueeze(0).expand(gen_num, -1).to(model.device)
        no_info_act = no_info_act.unsqueeze(0).expand(gen_num, -1).to(model.device)
        act_site_id = torch.full((gen_num,), model.probed_acts.index(probed_act), device=model.device)

        save_dir_path = os.path.join(args.save_generation_dir, "".join(q_input))
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        for temperature in [0.5, 1.0, 2.0, 4.0]:
            generated_tokens = model.generate(act, act_site_id, gen_mode, temperature=temperature)

            # insert manual samples here
            # manual_example = [
            #                 # [10, 6, 5, 6, 11, 1, 8, 0, 12, 8, 3, 6, 13, 14],
            #                 # [10, 1, 5, 6, 11, 6, 8, 0, 12, 8, 3, 6, 13, 14],
            #                 # [10, 3, 5, 6, 11, 4, 8, 0, 12, 8, 3, 6, 13, 14],
            #                 # [10, 4, 5, 6, 11, 3, 8, 0, 12, 8, 3, 6, 13, 14],
            #                 [10, 5, 3, 6, 11, 2, 8, 0, 12, 8, 1, 6, 13, 14],
            #                 [10, 5, 2, 6, 11, 2, 8, 0, 12, 8, 0, 6, 13, 14],
            #                 [10, 5, 1, 6, 11, 2, 8, 0, 12, 7, 9, 6, 13, 14],
            #                 [10, 5, 0, 6, 11, 2, 8, 0, 12, 7, 8, 6, 13, 14],
            #                 ]
            # manual_example = torch.LongTensor(manual_example).to(model.device)
            # generated_tokens[-manual_example.size(0):, :] = manual_example[:, :generated_tokens.size(1)]

            lm_logits = model(generated_tokens, act, act_site_id, use_cache=False)[0]
            lm_logits_baseline = model(generated_tokens, no_info_act, act_site_id, use_cache=False)[0]

            prob = torch.gather(F.softmax(lm_logits[:, :-1], dim=-1), dim=2, index=generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            prob_baseline = torch.gather(F.softmax(lm_logits_baseline[:, :-1], dim=-1), dim=2, index=generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            prob_diff = prob - prob_baseline    # -1.0  1.0
            prob_diff = torch.hstack([torch.zeros(gen_num, 1, device=model.device), prob_diff]) # for bos

            cache = add_necessary_hooks(hooked_model, [probed_act])
            hooked_model(generated_tokens, return_type=None)
            recomputed_act = retrieve_act(probed_act, cache)
            end_caching(hooked_model)

            pad_mask = (generated_tokens==tokenizer.pad_token_id) | (generated_tokens==tokenizer.eos_token_id)

            for metric in ["sim", "dist"]:
                if metric == "sim":
                    sim = F.cosine_similarity(act.unsqueeze(1), recomputed_act, dim=-1)
                    sim.masked_fill_(pad_mask, -1.0)
                    best_values, best_indices = sim.max(dim=1)
                    metric_values = sim
                elif metric == "dist":
                    diff = torch.linalg.vector_norm(act.unsqueeze(1) - recomputed_act, dim=-1) / torch.linalg.vector_norm(act.unsqueeze(1), dim=-1)
                    diff.masked_fill_(pad_mask, 1e3)
                    best_values, best_indices = diff.min(dim=1)
                    metric_values = diff

                items = []
                cache_obj = {"query_input": q_input, "query_position": q_pos, "probed_act": probed_act, "attn_weights": attn_w, "temperature": temperature, "metric": metric}
                for j in range(generated_tokens.size(0)):
                    
                    m = generated_tokens[j] != tokenizer.pad_token_id
                    filtered_t = generated_tokens[j][m].tolist()
                    filtered_p = [round(p, 3) for p in prob_diff[j][m].tolist()]
                    filtered_m = [round(v, 3) for v in metric_values[j][m].tolist()]
                    items.append( (convert_to_str_tokens(tokenizer, filtered_t), filtered_p, filtered_m, best_indices[j].item(), round(best_values[j].item(), 3)) )

                items = sorted(items, key=lambda x: -x[-1] if metric == "sim" else x[-1])
                cache_obj["generation"] = items

                save_path = os.path.join(save_dir_path, f"{probed_act}-{q_pos}-{temperature}-{metric}.json")
                with open(save_path, "w") as f:
                    json.dump(cache_obj, f)


def convert_to_freq(args, generated_tokens: list[torch.LongTensor], tokenizer: PreTrainedTokenizerFast, low_freq_mask: torch.BoolTensor):
    if (len(generated_tokens) == 0) or (args.probed_task != "fact"):
        return None
    vocab_len = len(tokenizer)
    max_seq_len = max(tokens_batch.size(1) for tokens_batch in generated_tokens)

    freqs = []
    for tokens_batch in generated_tokens:
        freq = (tokens_batch.unsqueeze(-1) == torch.arange(vocab_len, device=tokens_batch.device, dtype=torch.long).view(1, 1, -1)).sum(dim=0)
        freq = torch.cat([freq, torch.zeros(max_seq_len-freq.size(0), vocab_len, device=freq.device, dtype=freq.dtype)], dim=0)
        freqs.append(freq)
    freqs = torch.stack(freqs).sum(dim=0)
    freqs = ((freqs > 5) & low_freq_mask.unsqueeze(0).to(tokens_batch.device)).float()
    freqs[:, tokenizer.eos_token_id] = 0
    return freqs

def rand_cos_sim(v: torch.FloatTensor, costheta: float, num: int):
    assert v.dim() == 1
    u = v / torch.linalg.vector_norm(v)

    r = torch.randn(num, v.size(0), device=v.device)

    # Form a vector perpendicular to v:
    uperp = r - torch.matmul(r, u.unsqueeze(1)) * u.unsqueeze(0)
    uperp = uperp / torch.linalg.vector_norm(uperp, dim=-1, keepdim=True)

    # w is the linear combination of u and uperp with coefficients costheta
    w = costheta * u.unsqueeze(0) + math.sqrt(1 - costheta**2) * uperp

    mag = torch.linalg.vector_norm(v) * (torch.rand((num, 1), device=v.device)*2-1).exp()
    w = w * mag
    
    return w

@torch.no_grad()
def generate_auto_temp(
        args,
        test_act: list[tuple[torch.FloatTensor, str, str, torch.FloatTensor]], 
        model: CustomGPT2LMHeadModel, 
        tokenizer: PreTrainedTokenizerFast, 
        hooked_model: HookedTransformer,
    ):
    gen_mode = "sample"
    candidate_num = 64
    gen_num = 100
    if args.probed_task != "fact":
        subset_threshold = 0.1 
        low_freq_mask = None 
    else:
        subset_threshold = 0.25
        with open("../factual/token_freq.json", "r") as f:
            token_freq = torch.tensor(json.load(f))
            low_freq_mask = token_freq < 50000
            low_freq_mask = torch.cat([low_freq_mask, torch.zeros((2,), dtype=torch.bool)], dim=0)
    
    print("generating...")
    for i, (orig_act, probed_act, no_info_act, q_input, q_pos, attn_w) in enumerate(tqdm(test_act)):
        block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))

        save_dir_path = os.path.join(args.save_generation_dir, "".join(q_input))
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        no_info_act = no_info_act.unsqueeze(0).expand(gen_num, -1).to(model.device)

        generated_tokens = []
        if args.probed_task == "ioi":   # "fact"
            temp_range = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.1)] * 4
        elif args.probed_task == "fact":
            temp_range = [(1.0, 0.1), (1.0, 0.15), (1.0, 0.2), (1.0, 0.25)] * 4
            temp_range = list(reversed(temp_range))
        else:
            temp_range = [(0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.1)] * 4
        for temperature, noise_scalor in temp_range:
            if args.probed_task == "fact":
                act = rand_cos_sim(orig_act, 1 - noise_scalor, candidate_num)
            else:
                act = orig_act.unsqueeze(0).expand(candidate_num, -1).to(model.device)
                act = act + (torch.randn_like(act) * act[0].std() * noise_scalor)
            act_site_id = torch.full((candidate_num,), model.probed_acts.index(probed_act), device=model.device)
            generated_tokens.append(model.generate(act, act_site_id, gen_mode, args.max_len, temperature=temperature, prev_freq=convert_to_freq(args, generated_tokens, tokenizer, low_freq_mask) )) # 
        max_len = max(t.size(1) for t in generated_tokens)
        generated_tokens = [torch.cat([t, torch.full((candidate_num, max_len-t.size(1)), tokenizer.pad_token_id, device=t.device, dtype=t.dtype)], dim=1) for t in generated_tokens]
        generated_tokens = torch.cat(generated_tokens, dim=0)
        assert generated_tokens.size(0) == candidate_num * len(temp_range)

        cache = add_necessary_hooks(hooked_model, [probed_act])
        recomputed_act = []
        for j in range(0, len(generated_tokens), args.recompute_bz):
            hooked_model(generated_tokens[j:j+args.recompute_bz], stop_at_layer=block_idx+1)
            recomputed_act.append(retrieve_act(probed_act, cache))
        recomputed_act = torch.cat(recomputed_act, dim=0)
        end_caching(hooked_model)

        pad_mask = (generated_tokens==tokenizer.pad_token_id) | (generated_tokens==tokenizer.eos_token_id)
            
        act = orig_act.unsqueeze(0).expand(candidate_num*len(temp_range), -1).to(model.device)

        for metric in ["sim", "dist"]:
            save_path = os.path.join(save_dir_path, f"{probed_act}-{q_pos}-Auto-{metric}.json")
            # if os.path.exists(save_path):
            #     continue
            
            if metric == "sim":
                sim = F.cosine_similarity(act.unsqueeze(1), recomputed_act, dim=-1)
                sim.masked_fill_(pad_mask, -1.0)
                best_values, best_indices = sim.max(dim=1)
                metric_values = sim

                good_mask = (1 - best_values) <= subset_threshold

            elif metric == "dist":
                diff = torch.linalg.vector_norm(act.unsqueeze(1) - recomputed_act, dim=-1) / torch.linalg.vector_norm(act.unsqueeze(1), dim=-1)
                diff.masked_fill_(pad_mask, 1e3)
                best_values, best_indices = diff.min(dim=1)
                metric_values = diff

                good_mask = best_values <= subset_threshold

            weight = 1.0
            for j in range(25):
                indices = torch.multinomial(torch.where(good_mask, weight, 1.0), gen_num)
                r = good_mask[indices].sum().item() / gen_num
                if r < 0.6:
                    weight *= 1.2
                elif r > 0.8:
                    weight /= 1.1
                else:
                    break
            # else:
            #     print("warning: bad ratio", r)
            best_values = best_values[indices]
            best_indices = best_indices[indices]
            metric_values = metric_values[indices]
            g_tokens = generated_tokens[indices]

            act_site_id = torch.full((gen_num,), model.probed_acts.index(probed_act), device=model.device)
            lm_logits = model(g_tokens, act[indices], act_site_id, use_cache=False)[0]
            lm_logits_baseline = model(g_tokens, no_info_act, act_site_id, use_cache=False)[0]

            prob = torch.gather(F.softmax(lm_logits[:, :-1], dim=-1), dim=2, index=g_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            prob_baseline = torch.gather(F.softmax(lm_logits_baseline[:, :-1], dim=-1), dim=2, index=g_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            prob_diff = prob - prob_baseline    # -1.0  1.0
            prob_diff = torch.hstack([torch.zeros(gen_num, 1, device=model.device), prob_diff]) # for bos
  

            items = []
            cache_obj = {"query_input": q_input, "query_position": q_pos, "probed_act": probed_act, "attn_weights": attn_w, "temperature": 0.0, "metric": metric}
            for j in range(g_tokens.size(0)):
                
                m = g_tokens[j] != tokenizer.pad_token_id
                filtered_t = g_tokens[j][m].tolist()
                filtered_p = [round(p, 3) for p in prob_diff[j][m].tolist()]
                filtered_m = [round(v, 3) for v in metric_values[j][m].tolist()]
                items.append( (convert_to_str_tokens(tokenizer, filtered_t), filtered_p, filtered_m, best_indices[j].item(), round(best_values[j].item(), 3)) )

            items = sorted(items, key=lambda x: -x[-1] if metric == "sim" else x[-1])
            cache_obj["generation"] = items

            with open(save_path, "w") as f:
                json.dump(cache_obj, f)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probed_task", type=str, choices=["addition", "counting", "ioi", "fact"])
    parser.add_argument("--recompute_bz", type=int, default=100)
    parser.add_argument("--no_auto", action="store_true")
    args = parser.parse_args()

    if args.probed_task == "addition":
        args.save_dir = "addition"
        args.save_generation_dir = "../training_outputs/cached_addition_generation"
    elif args.probed_task == "counting":
        args.save_dir = "counting"
        args.save_generation_dir = "../training_outputs/cached_counting_generation"
    elif args.probed_task == "ioi":
        args.save_dir = "ioi"
        args.save_generation_dir = "../training_outputs/cached_ioi_generation"
    elif args.probed_task == "fact":
        args.save_dir = "fact"
        args.save_generation_dir = "../training_outputs/cached_fact_generation"
  

    torch.manual_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    print(args)

    probed_model_path, data_path, decoder_dir, max_len = get_paths(args)
    args.max_len = max_len

    if args.probed_task not in  ["addition", "counting"]:
        tokenizer = AutoTokenizer.from_pretrained(probed_model_path, add_bos_token=True)
    else:
        tokenizer = None
    probed_model = GPT2LMHeadModel.from_pretrained(probed_model_path)

    if args.probed_task in ["ioi", "fact"]:
        tokenizer.add_special_tokens({"eos_token": "[EOS]", "pad_token": "[PAD]"})
        #  bos: <|endoftext|> 50256     eos: [EOS] 50257     pad: [PAD] 50258
        probed_model.resize_token_embeddings(probed_model.config.vocab_size+2)
        probed_model.config.eos_token_id = tokenizer.eos_token_id
        probed_model.config.pad_token_id = tokenizer.pad_token_id
        
    hooked_model = HookedTransformer.from_pretrained(
            "gpt2",
            hf_model=probed_model,
            device=device,
            tokenizer=tokenizer,
            n_embd=probed_model.config.n_embd,
            n_layer=probed_model.config.n_layer,
            n_head=probed_model.config.n_head,
            vocab_size=probed_model.config.vocab_size,
            n_ctx=probed_model.config.n_positions,

    )
    
    print(hooked_model.embed.W_E.device)
    if args.probed_task == "addition":
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from addition.train import customTokenizer, make_dataset
        tokenizer = customTokenizer()
        hooked_model.tokenizer = tokenizer
    elif args.probed_task == "counting":
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from counting.train import customTokenizer, make_dataset
        tokenizer = customTokenizer()
        hooked_model.tokenizer = tokenizer
    hooked_model.eval()
    del probed_model

    checkpoint_paths = glob("checkpoint-*", root_dir=decoder_dir)
    n = max([int(re.search(r"\d+", p).group(0)) for p in checkpoint_paths])
    last_checkpoint_path = os.path.join(decoder_dir, f"checkpoint-{n}")

    print("loading from dir:", last_checkpoint_path)
    model = CustomGPT2LMHeadModel.from_pretrained(last_checkpoint_path).to(device)
    model.eval()
 
    
    if args.probed_task == "ioi":
        with open(data_path, "rb") as f:
            rollouts: list[dict[str, str]] = pickle.load(f)
        # {'[PLACE]': 'hospital', '[OBJECT]': 'basketball', 'text': 'The local big hospital Jacob and Jason went to had a basketball. Jason gave it to Jacob', 'IO': 'Jacob', 'S': 'Jason', 'TEMPLATE_IDX': 11, 'pos_IO': 4, 'pos_IO-1': 3, 'pos_IO+1': 5, 'pos_S': 6, 'pos_S-1': 5, 'pos_S+1': 7, 'pos_S2': 13, 'pos_end': 16, 'pos_starts': 0, 'pos_punct': 12}
        def check_if_ioi(obj):
            tokens = obj["text"].split(" ")
            return (tokens.count(obj["IO"]) == 2) and (tokens.count(obj["S"]) == 2)
        
        rollouts = list(filter(check_if_ioi, rollouts))

        random_idx = random.sample(list(range(len(rollouts))), 50)

        test_rollouts = []
        test_indices = []
        for i in random_idx:
            obj = rollouts[i]
            test_rollouts.append(obj["text"])
            # important pos: end, s2, s1+1
            test_indices.append((obj["pos_S+1"]+1, obj["pos_S2"]+1, obj["pos_end"]+1)) # +1 because of BOS

        probed_heads = "7.3 7.9 8.6 8.10 10.7 11.10 9.9 9.6 10.0 9.0 9.7 10.1 10.2 10.6 10.10 11.2 11.9 0.1 3.0 0.10 5.5 6.9 5.8 5.9 2.2 4.11"
        probed_acts = []
        for head in probed_heads.split(" "):
            layer_idx, head_idx = head.split(".")
            probed_acts.append(f"blocks.{layer_idx}.attn.hook_result.{head_idx}")

    elif args.probed_task == "addition":
        test_rollouts = [
            "615+861=1476",
            "925+398=1323",
            "101+539=640",
            "556+280=836",
            "271+829=1100",
            "403+288=691",
            "715+916=1631",

            "573+269=842",
            "352+106=458",
            "711+391=1102",
            "835+141=976",
            "638+152=790",
            "278+286=564",  # make sure those shown in the paper are included
        ]
        train_dataset, _ = make_dataset(tokenizer, 1.0)

        existing_pairs = [(int(item[:3]), int(item[4:7])) for item in test_rollouts]
        for i in range(200-len(test_rollouts)):
            pair = train_dataset.data[random.randint(0, len(train_dataset)-1)]
            while pair in existing_pairs:
                pair = train_dataset.data[random.randint(0, len(train_dataset)-1)]

            a, b = pair
            s = str(a) + "+" + str(b) + "=" + str(a+b)
            test_rollouts.append(s)
        
        test_indices = [(4, 8, 9, 10, 11)] * len(test_rollouts)

        probed_acts = model.probed_acts

    elif args.probed_task == "counting":
        train_dataset, _ = make_dataset(tokenizer, 1.0)

        random_idx = random.sample(list(range(len(train_dataset))), 200)

        test_rollouts = []
        for i in random_idx:
            s = train_dataset.data[i]
            test_rollouts.append(s)
        
        test_indices = [(-2, -3, -4)] * len(test_rollouts)
    
        probed_acts = model.probed_acts

    elif args.probed_task == "fact":
        def find_subseq(sequence: list[int], subseq: list[int]):
            for i in range(len(sequence)-len(subseq)+1):
                if tuple(sequence[i:i+len(subseq)]) == tuple(subseq):
                    return i
            print(tokenizer.convert_ids_to_tokens(sequence))
            print(tokenizer.convert_ids_to_tokens(subseq))
            return 0

        with open("../datasets/factual/known_1000.json", "r") as f:
            examples = json.load(f)
        test_rollouts = []
        test_indices = []

        for item in examples[:1]:     
            text = item["prompt"] + " " + item["attribute"]
            relation_idx =  -1 * (len(tokenizer(" "+item["attribute"])["input_ids"]) + 1)
            
            subj_subseq = item["subject"] if item["template"].startswith("{") else " "+item["subject"]
            subj_subseq = tokenizer(subj_subseq)["input_ids"]
            text_seq = tokenizer(text)["input_ids"]
            subj_idx = find_subseq(text_seq, subj_subseq) + len(subj_subseq) # there is a bos

            test_rollouts.append(text)
            test_indices.append((subj_idx, relation_idx,))   # 

        probed_acts = model.probed_acts

    print("capturing activation...")
    test_act = []  
    for rollout, indices in tqdm(zip(test_rollouts, test_indices)):
        test_act.extend( get_test_activation(probed_acts, hooked_model, rollout, indices) )

    if args.no_auto:
        generate_and_print(args, test_act, model, tokenizer, hooked_model)
    else:
        generate_auto_temp(args, test_act, model, tokenizer, hooked_model)

        
                
