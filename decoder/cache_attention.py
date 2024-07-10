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

from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer

from utils import *
from model import CustomGPT2LMHeadModel
from cache_generation import convert_to_str_tokens


# logging.basicConfig(level=logging.INFO)
   

def cache_attention(args, hooked_model: HookedTransformer, text: list[str]):
    n_layers = hooked_model.cfg.n_layers
    n_heads = hooked_model.cfg.n_heads
    cache = add_necessary_hooks(hooked_model, [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)])
    tokens = hooked_model.to_tokens(text)
    with torch.no_grad():
        hooked_model(tokens, return_type=None)

    for batch_idx in range(len(text)):
        num_pad = ((tokens[batch_idx] == hooked_model.tokenizer.pad_token_id) | (tokens[batch_idx] == hooked_model.tokenizer.eos_token_id)).sum().item()
        if num_pad > 0:
            one_input = tokens[batch_idx, :-num_pad]
        else:
            one_input = tokens[batch_idx]
        one_input = convert_to_str_tokens(hooked_model.tokenizer, one_input.tolist())

        save_dir_path = os.path.join(args.save_attention_dir, "".join(one_input))
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        for i in range(n_layers):
            for j in range(n_heads):
                # [batch, head_index, query_pos, key_pos]
                if num_pad > 0:
                    attn_weights = cache[f"blocks.{i}.attn.hook_pattern"][batch_idx, j, :-num_pad, :-num_pad]
                else:
                    attn_weights = cache[f"blocks.{i}.attn.hook_pattern"][batch_idx, j]
                attn_weights = attn_weights.tolist()
                attn_weights = [list(map(lambda w:round(w, 3), row)) for row in attn_weights]
                
                cache_obj = {"str_tokens": one_input, "layer_idx": i, "head_idx": j, "attn_weights": attn_weights}

                save_path = os.path.join(save_dir_path, f"{i}-{j}.json")
                with open(save_path, "w") as f:
                    json.dump(cache_obj, f)

    end_caching(hooked_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probed_task", type=str, choices=["addition", "counting"], default="addition")

    args = parser.parse_args()

    if args.probed_task == "addition":
        args.save_dir = "addition"
        args.save_generation_dir = "../training_outputs/cached_addition_generation"
    elif args.probed_task == "counting":
        args.save_dir = "counting"
        args.save_generation_dir = "../training_outputs/cached_counting_generation"
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(args)

    probed_model_path, data_path, _, max_len = get_paths(args)

    if args.probed_task not in ["addition", "counting"]:
        tokenizer = AutoTokenizer.from_pretrained(probed_model_path, add_bos_token=True)
    else:
        tokenizer = None
    probed_model = GPT2LMHeadModel.from_pretrained(probed_model_path)

    if args.probed_task == "ioi":
        tokenizer.add_special_tokens({"eos_token": "[EOS]", "pad_token": "[PAD]"})
        #  bos: <|endoftext|> 50256     eos: [EOS] 50257     pad: [PAD] 50258
        probed_model.resize_token_embeddings(probed_model.config.vocab_size+2)
        probed_model.config.eos_token_id = tokenizer.eos_token_id
        probed_model.config.pad_token_id = tokenizer.pad_token_id
        
    hooked_model = HookedTransformer.from_pretrained(
            "gpt2",
            hf_model=probed_model,
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
 
    
    if args.probed_task == "addition":
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
        
    elif args.probed_task == "counting":
        train_dataset, _ = make_dataset(tokenizer, 1.0)

        random_idx = random.sample(list(range(len(train_dataset))), 200)

        test_rollouts = []
        for i in random_idx:
            s = train_dataset.data[i]
            test_rollouts.append(s)

    batch_size = 32
    print("capturing attention...")
    for s in range(0, len(test_rollouts), batch_size):
        cache_attention(args, hooked_model, test_rollouts[s: s+batch_size])
        
                
