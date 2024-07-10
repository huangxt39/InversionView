import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import re
from collections import OrderedDict
import os
import random
import json
import sys
from tqdm import tqdm
import math

from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, HookedTransformerConfig
from model import CustomGPT2LMHeadModel
from utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from addition.train import customTokenizer, make_dataset

torch.set_grad_enabled(False)
torch.set_printoptions(sci_mode=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def calculate_dist_and_prob(save_path, selected_pos, seed=0, temp=1.0, noise=None):
    torch.manual_seed(seed)

    probed_model_path = "../training_outputs/addition_fixed/checkpoint-59350"

    probed_model = GPT2LMHeadModel.from_pretrained(probed_model_path)
    tokenizer = customTokenizer()

    hooked_model = HookedTransformer.from_pretrained(
            "gpt2",
            hf_model=probed_model,
            tokenizer=None,
            n_embd=probed_model.config.n_embd,
            n_layer=probed_model.config.n_layer,
            n_head=probed_model.config.n_head,
            vocab_size=probed_model.config.vocab_size,
            n_positions=probed_model.config.n_positions,
            n_ctx=probed_model.config.n_positions,
    )


    hooked_model.eval()
    del probed_model
    hooked_model.tokenizer = tokenizer
    # hooked_model.cpu()
    print(hooked_model.embed.W_E.device)

    decoder_dir = "data_and_model/addition/checkpoint-195400"
    model = CustomGPT2LMHeadModel.from_pretrained(decoder_dir).to(device)
    model.eval()

    dataset, _ = make_dataset(tokenizer, 1.0)

    rand_idx = sorted(torch.randperm(len(model.probed_acts)-1)[:8].tolist())
    # print([model.probed_acts[n] for n in rand_idx ])
    # exit()
    act_location = []
    for n in rand_idx:
        probed_act = model.probed_acts[n+1]
        sel_pos = selected_pos[torch.randint(0, len(selected_pos), ()).item()]
        act_location.append((probed_act, sel_pos))
    print(act_location)
    for probed_act, sel_pos in act_location:
        block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))

        query_input = torch.LongTensor([dataset[ torch.randint(0, len(dataset), ()) ]])
        query_string = tokenizer.convert_ids_to_tokens(query_input[0].tolist())
        query_string[sel_pos] = "(" + query_string[sel_pos] + ")"
        query_string = "".join(query_string)
        
        cache = add_necessary_hooks(hooked_model, [probed_act])
        hooked_model(query_input, stop_at_layer=block_idx+1)

        query_act = retrieve_act(probed_act, cache)[0, sel_pos].clone() 

        # distance for all
        batch_size = 1024
        print("calculating distance")
        all_dist = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_input = torch.LongTensor([dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]).to(device)
            hooked_model(batch_input, stop_at_layer=block_idx+1)
            recomputed_act = retrieve_act(probed_act, cache).clone() 

            pad_mask = (batch_input==tokenizer.pad_token_id) | (batch_input==tokenizer.eos_token_id)

            diff = torch.linalg.vector_norm(query_act.view(1, 1, -1) - recomputed_act, dim=-1) / torch.linalg.vector_norm(query_act, dim=-1)
            diff.masked_fill_(pad_mask, 1e3)
            best_values, best_indices = diff.min(dim=1)
            all_dist.append(best_values)
        all_dist = torch.cat(all_dist, dim=0)
        end_caching(hooked_model)
        print(all_dist.size())

        # prob for all
        print("calculating probability")
        if noise is None:
            all_prob = []
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch_input = torch.LongTensor([dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]).to(device)
                act_site_id = torch.full((batch_input.size(0),), model.probed_acts.index(probed_act), device=model.device)
                logits = model(batch_input, query_act.unsqueeze(0).expand(batch_input.size(0), -1), act_site_id, use_cache=False)[0]
                log_prob = F.log_softmax(logits[:, :-1] / temp, dim=-1)
                log_prob = torch.gather(log_prob, dim=2, index=batch_input[:, 1:].unsqueeze(-1)).squeeze(-1)

                log_prob.masked_fill_(batch_input[:, 1:] == tokenizer.pad_token_id, 0)
                log_prob = log_prob.sum(dim=-1)
                all_prob.append(log_prob)

            all_prob = torch.cat(all_prob, dim=0)
            print(all_prob.size())
        
        else:
            all_prob = []
            n_samples = 500
            for k in tqdm(range(n_samples)):    # monte carlo estimate
                noisy_query_act = query_act + torch.randn_like(query_act) * query_act.std() * noise
                prob_per_sample = []
                for i in range(0, len(dataset), batch_size):
                    batch_input = torch.LongTensor([dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]).to(device)
                    act_site_id = torch.full((batch_input.size(0),), model.probed_acts.index(probed_act), device=model.device)
                    logits = model(batch_input, noisy_query_act.unsqueeze(0).expand(batch_input.size(0), -1), act_site_id, use_cache=False)[0]
                    log_prob = F.log_softmax(logits[:, :-1] / temp, dim=-1)
                    log_prob = torch.gather(log_prob, dim=2, index=batch_input[:, 1:].unsqueeze(-1)).squeeze(-1)

                    log_prob.masked_fill_(batch_input[:, 1:] == tokenizer.pad_token_id, 0)
                    log_prob = log_prob.sum(dim=-1)
                    prob_per_sample.append(log_prob)

                prob_per_sample = torch.cat(prob_per_sample, dim=0)
                all_prob.append(prob_per_sample)
            all_prob = torch.stack(all_prob)
            print(all_prob.size())
            all_prob = torch.logsumexp(all_prob, dim=0) - math.log(n_samples)

        new_item = {"probed_act": probed_act, "query_input": query_string, "selected_pos": sel_pos, \
                    "dist": all_dist.cpu(), "prob": all_prob.cpu()}
        if os.path.exists(save_path):
            existing_obj = torch.load(save_path)
            existing_obj.append(new_item)
            torch.save(existing_obj, save_path)
        else:
            torch.save([new_item], save_path)



save_path = "data_and_model/scatter_data_new_noise1.pt"
calculate_dist_and_prob(save_path, [8, 9, 10, 11], noise=0.1)

