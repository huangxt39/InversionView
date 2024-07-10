import torch
import torch.nn as nn
import torch.nn.functional as F
import circuitsvis as cv

import numpy as np
import re
from collections import OrderedDict
import os
import random
from tqdm import tqdm
import json

from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, HookedTransformerConfig
from functools import partial
from train import*



def filter_inputs(input_ids: torch.LongTensor):
    return input_ids

def filter_con_inputs(input_ids: torch.LongTensor):
    m = input_ids[:, 9] != 10000
    return m

def complete_ans(input_ids: torch.LongTensor):
    a, b, c = (input_ids[:, 1:4] + input_ids[:, 5:8]).split(1, dim=1)
    ans = (a*100 + b*10 + c).squeeze(-1)

    m = ans >= 1000

    temp = 0
    for i, divisor in enumerate([1000, 100, 10, 1]):
        input_ids[m, 9+i] = (ans[m] - temp) // divisor
        temp += (ans[m] - temp) // divisor * divisor
    assert (temp == ans[m]).all(0)
    input_ids[m, -1] = 13

    temp = 0
    for i, divisor in enumerate([100, 10, 1]):
        input_ids[~m, 9+i] = (ans[~m] - temp) // divisor
        temp += (ans[~m] - temp) // divisor * divisor
    assert (temp == ans[~m]).all()
    input_ids[~m, -2] = 13
    input_ids[~m, -1] = 14
    
    return input_ids



def make_contrast_inputs(input_ids: torch.LongTensor, target_pos: int, changed_pos: list[int]):
    batch_size = input_ids.size(0)
    device = input_ids.device

    con_input_ids = input_ids.clone()
    terminated = torch.zeros(batch_size, device=device, dtype=torch.bool)
    try:
        while not terminated.all().item():
            for pos in changed_pos:
                con_input_ids[~terminated, pos] = torch.randint(0, 10, ((~terminated).sum().item(),), device=device)
            con_input_ids = complete_ans(con_input_ids)

            terminated = con_input_ids[:, target_pos] != input_ids[:, target_pos]
            terminated = terminated & filter_con_inputs(con_input_ids)
    except KeyboardInterrupt:
        print(input_ids[~terminated])
        exit()
    
    return con_input_ids

def head_list_to_dict(heads: list[tuple[int, int]]):
    heads_per_layer = {0:[], 1:[]}
    for l, h in heads:
        heads_per_layer[l].append(h)
    return heads_per_layer



torch.set_grad_enabled(False)
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
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
hooked_model.set_use_attn_result(True)

train_dataset, test_dataset = make_dataset(tokenizer)

data_collator = customCollator(tokenizer.pad_token_id)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=data_collator)

def replace_attn_result(x, hook, head_idx: list[int], pos_idx: int, corrupted_act):
    # hook_result: [batch, pos, head_index, d_model]
    assert isinstance(pos_idx, int)
    x[:, pos_idx, head_idx] = corrupted_act[:, pos_idx, head_idx]
    return x

target_pos = 12  # A4
patched_pos = 11    # A3
changed_pos = [1, 5] # F1, S1 -> A1
results = {}
for j in range(2):
    if j == 0:
        all_heads = [(i, j) for i in range(2) for j in range(4)]
    else:
        all_heads = list(reversed([(i, j) for i in range(2) for j in range(4)]))
    diff_diff = []
    for i in range(len(all_heads)):
        patched_heads = all_heads[:i+1]
        patched_heads = head_list_to_dict(patched_heads)
        print("patching", patched_heads)
        avg_orig_diff = 0
        avg_corrupt_diff = 0
        total_num = 0
        for inputs in tqdm(test_loader):
            input_ids : torch.LongTensor = inputs["input_ids"].to(device)
            input_ids = filter_inputs(input_ids)
            if input_ids.size(0) == 0:
                continue
            con_input_ids = make_contrast_inputs(input_ids, target_pos, changed_pos)
            bz, seq_len = input_ids.size()

            orig_logits = hooked_model(input_ids, return_type="logits")
            
            hooked_model.reset_hooks()
            _, cache = hooked_model.run_with_cache(con_input_ids, return_type=None)
            
            fwd_hooks = []
            if patched_heads[0]:
                temp_hook1 = partial(replace_attn_result, head_idx=patched_heads[0], pos_idx=patched_pos, corrupted_act=cache["blocks.0.attn.hook_result"])
                fwd_hooks.append(("blocks.0.attn.hook_result", temp_hook1))

            if patched_heads[1]:
                temp_hook2 = partial(replace_attn_result, head_idx=patched_heads[1], pos_idx=patched_pos, corrupted_act=cache["blocks.1.attn.hook_result"])
                fwd_hooks.append(("blocks.1.attn.hook_result", temp_hook2))

            corrupt_logits = hooked_model.run_with_hooks(input_ids, 
                                return_type="logits",
                                fwd_hooks=fwd_hooks)

            orig_label = input_ids[:, target_pos]
            con_label = con_input_ids[:, target_pos]
            arange_idx = torch.arange(bz, device=device, dtype=torch.long)
            orig_diff = orig_logits[arange_idx, target_pos-1, orig_label] - orig_logits[arange_idx, target_pos-1, con_label]
            corrupt_diff = corrupt_logits[arange_idx, target_pos-1, orig_label] - corrupt_logits[arange_idx, target_pos-1, con_label]

            avg_orig_diff += orig_diff.sum().item()
            avg_corrupt_diff += corrupt_diff.sum().item()
            total_num += bz

        avg_orig_diff /= total_num
        avg_corrupt_diff /= total_num

        print(avg_orig_diff)
        print(avg_corrupt_diff)
        diff_diff.append(avg_orig_diff - avg_corrupt_diff)

    if j == 0:
        results["forward"] = {"heads": all_heads, "values": diff_diff}
    else:
        results["backward"] = {"heads": all_heads, "values": diff_diff}

with open("causal_exp.json", 'w') as f:
    json.dump(results, f)

print(results)


# figure a
# target_pos = 9  # A1
# patched_pos = 8   # = 
# changed_pos = [1, 5] # F1, S1
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [8.517165224588652, 18.895440629238553, 24.563945439060824, 24.599568918619035, 25.124327847939952, 25.12175770423795, 25.143308308862757, 25.30910744739344]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [0.48993402883270676, 0.4551087351104357, 4.742325250809282, 5.451862936231825, 5.46957524635409, 8.919526337687175, 18.86212242493806, 25.255155515732]}}

# target_pos = 9  # A1
# patched_pos = 8   # = 
# changed_pos = [2, 6] # F2, S2
# def filter_inputs(input_ids: torch.LongTensor):
#     m = (input_ids[:, 1] + input_ids[:, 5]) < 10
#     return input_ids[m].clone()
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [0.027509578003011015, 0.026998206465060903, 0.02660437968489049, 13.254323847800647, 13.35283056354993, 13.424196300260395, 13.415576400215834, 13.470630031630282]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [-0.13969961156555044, -0.1700980371850651, 1.3860414132009442, 1.8459712914869488, 13.45411301068101, 13.452674381768999, 13.457144155064341, 13.461305483648516]}}

# figure b
# target_pos = 10  # A2
# patched_pos = 9    # A1
# changed_pos = [1, 5] # F1, S1
# def filter_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [0.0002829056832052146, 0.0025898564684254666, 4.571424759869441, 5.939075379840704, 27.23797240417881, 27.50385161009538, 27.62147048800296, 28.833387663443702]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [1.7328973352301613, 1.823556919947558, 2.2089118728099137, 23.548902820974888, 24.39737647055628, 28.849406812317653, 28.854166214591725, 28.80235773625393]}}


# target_pos = 10  # A2
# patched_pos = 9    # A1
# changed_pos = [2, 6] # F2, S2
# def filter_inputs(input_ids: torch.LongTensor):
#     m = (input_ids[:, 1] + input_ids[:, 5]) >= 10
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [-0.0009417391687867038, 0.0010888755972002784, 0.015685946598654077, 0.1563877597397445, 8.873911507537812, 9.578198228771742, 10.211350442797226, 11.735800090361408]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [1.5070384372352805, 2.104248958483821, 2.7892900640584615, 11.542720382340718, 11.699278266983544, 11.745034052809604, 11.732989456726983, 11.729264963975787]}}


# figure c
# target_pos = 10  # A2
# patched_pos = 9    # A1
# changed_pos = [2, 6] # F2, S2
# def filter_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] != 1
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] != 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [0.005960805583143269, 13.801098154753628, 29.524930464841994, 29.434018332708114, 33.69821596458189, 33.91153154850208, 34.823266046114895, 35.293102650123885]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [0.5490855631816309, 1.8001172825004428, 1.9788411740389407, 4.907531182513951, 4.90456338988027, 21.012282508286493, 35.19810640063501, 35.22451146763771]}}

# target_pos = 10  # A2
# patched_pos = 9    # A1
# changed_pos = [3, 7] # F3, S3
# def filter_inputs(input_ids: torch.LongTensor):
#     m = (input_ids[:, 9] != 1) & ((input_ids[:, 1] + input_ids[:, 5] != 9) | (input_ids[:, 2] + input_ids[:, 6] != 9))
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] != 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [0.005379941677520428, 0.001977674850150102, 0.0020292199519849063, 15.037974436840944, 15.057734787015443, 15.148835461431062, 15.18854869765341, 15.182328036848943]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [0.38696067093393527, 0.4664754701806242, 0.7899401618286044, 0.6404298251041265, 15.175744057255757, 15.1706478461907, 15.169873702256872, 15.182210259829525]}}


# figure d
# target_pos = 11  # A3
# patched_pos = 10    # A2
# changed_pos = [2, 6] # F2, S2
# def filter_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [0.004704056236704446, 14.446088708548714, 29.155840009971122, 29.12559118012981, 33.213755120946665, 33.65585049543512, 34.888159774498696, 35.21156561801561]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [0.519950433039277, 2.2149993759440605, 2.408746502514779, 5.292537640251405, 5.260238416769013, 20.40032261278791, 35.19030207340461, 35.26580676913587]}}

# target_pos = 11  # A3
# patched_pos = 10    # A2
# changed_pos = [3, 7] # F3, S3
# def filter_inputs(input_ids: torch.LongTensor):
#     m = (input_ids[:, 9] == 1) & ((input_ids[:, 1] + input_ids[:, 5] != 9) | (input_ids[:, 2] + input_ids[:, 6] != 9))
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [-8.494509446066445e-05, -0.000637357256487725, -0.0010570421460291968, 14.969253758142788, 14.97833529344002, 15.045270206642352, 15.083458314519415, 15.085392629426757]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [0.3296234180573787, 0.4183580552402342, 0.7156351165267267, 0.5669910925663295, 15.088374477423315, 15.087860069935665, 15.086279815435802, 15.086145135359708]}}


# figure e
# target_pos = 11  # A3
# patched_pos = 10    # A2
# changed_pos = [3, 7] # F3, S3
# def filter_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] != 1
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] != 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [-0.0014089303356215055, -0.0014276551341083632, -0.0017623198762386494, 1.9959713767070042, 2.81935232682679, 4.796778589975077, 18.953826072722723, 35.84909097591503]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [16.919811888961377, 31.04188248127327, 32.77227775254849, 33.557656909367516, 35.92080271415482, 35.87522924996663, 35.8676086763583, 35.922029562409804]}}


# figure f
# target_pos = 12  # A4
# patched_pos = 11    # A3
# changed_pos = [3, 7] # F3, S3
# def filter_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return input_ids[m].clone()
# def filter_con_inputs(input_ids: torch.LongTensor):
#     m = input_ids[:, 9] == 1
#     return m
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [-1.9991199131652593e-05, 1.3810264809505632, 1.9968409445830027, 4.003503319442009, 5.034646968101104, 6.808616071224236, 21.478738773190305, 37.350440168504775]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [15.866359063253544, 30.38138489593167, 30.82171984212036, 31.481932447724503, 34.71425512880084, 35.608731115521756, 37.31407471395208, 37.35823991688051]}}

# target_pos = 12  # A4
# patched_pos = 11    # A3
# changed_pos = [1, 5] # F1, S1 -> A1
# {'forward': {'heads': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 'values': [38.51468480801053, 38.50872252170892, 38.50375321903935, 38.52791429918077, 38.626071560857326, 38.77775327834141, 38.784715952291606, 38.79566357152491]}, 'backward': {'heads': [(1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (0, 2), (0, 1), (0, 0)], 'values': [-0.007767616592218474, -0.027007276332525976, 7.962263913415978, 8.441564627659172, 8.47005752056734, 8.45936695115831, 8.473442410108778, 38.802275314971546]}}
