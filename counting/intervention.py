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

torch.set_grad_enabled(False)
torch.set_printoptions(sci_mode=False)

def filter_inputs(input_ids: torch.LongTensor, tokenizer):
    eos_pos = (input_ids == tokenizer.eos_token_id).float().argmax(dim=-1)
    m = (eos_pos != 8) & (eos_pos != 27+5)
    return input_ids[m].clone()

def make_contrast_inputs(input_ids: torch.LongTensor, dataset: CharacterNum):
    bz = input_ids.size(0)
    device = input_ids.device

    def ans_equal(a, b):
        a, b = a.tolist(), b.tolist()
        a_ans = a[a.index(dataset.tokenizer.eos_token_id)-1]
        b_ans = b[b.index(dataset.tokenizer.eos_token_id)-1]
        return a_ans == b_ans

    # def ans_in_neighbour(a, c):
    #     a, c = a.tolist(), c.tolist()
    #     a_ans = int(dataset.tokenizer.vocab_inv[a[a.index(dataset.tokenizer.eos_token_id)-1]])
    #     c_ans = int(dataset.tokenizer.vocab_inv[c[c.index(dataset.tokenizer.eos_token_id)-1]])
    #     return (a_ans == c_ans - 1) or (a_ans == c_ans + 1)
    
    # def ans_in_neighbour_no_pos_signal(a, c):
    #     a, c = a.tolist(), c.tolist()
    #     a_len = a.index(dataset.tokenizer.eos_token_id)
    #     c_len = c.index(dataset.tokenizer.eos_token_id)
    #     a_ans = int(dataset.tokenizer.vocab_inv[a[a_len-1]])
    #     c_ans = int(dataset.tokenizer.vocab_inv[c[c_len-1]])

    #     if (a_ans == c_ans - 1) and (a_len >= c_len):
    #         return True
    #     elif (a_ans == c_ans + 1) and (a_len <= c_len):
    #         return True
    #     else:
    #         return False
    
    con_input_ids = []
    for i in range(bz):
        candidate = torch.LongTensor(dataset[random.randint(0, len(dataset)-1)]).to(device)
        while ans_equal(candidate, input_ids[i]): 
            candidate = torch.LongTensor(dataset[random.randint(0, len(dataset)-1)]).to(device)
        con_input_ids.append(candidate)

        # print(dataset.tokenizer.convert_ids_to_tokens(input_ids[i].tolist()))
        # print(dataset.tokenizer.convert_ids_to_tokens(candidate.tolist()))
        # print("=====")
    
    con_input_ids = torch.stack(con_input_ids)

    # verification
    arange_idx = torch.arange(bz, dtype=torch.long, device=device)
    ans_idx = (input_ids == dataset.tokenizer.eos_token_id).float().argmax(dim=-1) - 1
    ans_input_ids = input_ids[arange_idx, ans_idx]

    ans_idx = (con_input_ids == dataset.tokenizer.eos_token_id).float().argmax(dim=-1) - 1
    ans_con_input_ids = con_input_ids[arange_idx, ans_idx]
    
    assert (ans_input_ids != ans_con_input_ids).all()

    return con_input_ids

def flip_one_char(string, char_set):
    old_count = string[string.index(":") + 1]
    query_c = string[string.index("|") + 1]

    def check_validity(s, chars):
        for c in chars:
            if s.count(c) > 9 or s.count(c) < 1:
                return False
        return True

    while True:
        # flip
        rand_idx = random.randint(1, string.index("|")-1)
        assert string[rand_idx] in char_set
        char_set_temp = char_set.copy()
        char_set_temp.remove(string[rand_idx])
        string_temp = string.copy()
        string_temp[rand_idx] = random.choice(list(char_set_temp))
        
        # complete ans
        assert string_temp.count(query_c) > 0
        new_count = str(string_temp.count(query_c) - 1)
        if new_count != old_count and check_validity(string_temp[1:string_temp.index("|")], char_set):
            string_temp[string_temp.index(":") + 1] = new_count
            break

    return string_temp


def make_contrast_inputs_flip(input_ids: torch.LongTensor, dataset: CharacterNum):
    bz = input_ids.size(0)
    device = input_ids.device

    def ans_equal(a, b):
        a, b = a.tolist(), b.tolist()
        a_ans = a[a.index(dataset.tokenizer.eos_token_id)-1]
        b_ans = b[b.index(dataset.tokenizer.eos_token_id)-1]
        return a_ans == b_ans
    
    con_input_ids = []
    for i in range(bz):
        string = dataset.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
        chars = set(string[1:string.index("|")])
        string = flip_one_char(string, chars)
        
        candidate = torch.LongTensor(list(map(lambda x: dataset.tokenizer.vocab[x], string))).to(device)
        con_input_ids.append(candidate)

        # print(dataset.tokenizer.convert_ids_to_tokens(input_ids[i].tolist()))
        # print(dataset.tokenizer.convert_ids_to_tokens(candidate.tolist()))
        # print("=====")
    
    con_input_ids = torch.stack(con_input_ids)

    # verification
    arange_idx = torch.arange(bz, dtype=torch.long, device=device)
    ans_idx = (input_ids == dataset.tokenizer.eos_token_id).float().argmax(dim=-1) - 1
    ans_input_ids = input_ids[arange_idx, ans_idx]

    ans_idx = (con_input_ids == dataset.tokenizer.eos_token_id).float().argmax(dim=-1) - 1
    ans_con_input_ids = con_input_ids[arange_idx, ans_idx]
    
    assert (ans_input_ids != ans_con_input_ids).all()

    return con_input_ids


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
probed_model_path = "../training_outputs/counting/checkpoint-914100"

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

data_collator = customCollator(tokenizer.vocab[":"])
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=data_collator)

# Bvvzccvczvvvzvcvc|v:8EPPPPPPPPPPP


def replace_attn_result(x, hook, head_idx: int, pos_idx: torch.LongTensor, con_pos_idx: torch.LongTensor, corrupted_act):
    # hook_result: [batch, pos, head_index, d_model]
    arange_idx = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    x[arange_idx, pos_idx, head_idx] = corrupted_act[arange_idx, con_pos_idx, head_idx]
    return x

def replace_act(x, hook, pos_idx: torch.LongTensor, con_pos_idx: torch.LongTensor, corrupted_act):
    # resid: [batch, pos, d_model]
    arange_idx = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    x[arange_idx, pos_idx] = corrupted_act[arange_idx, con_pos_idx]
    return x


locations = [(-4, 0, 0), (-3, 0, 0), (-3, 1, 0)]  # (position index (E being the last token), layer, head)
patched_pos, patched_layer, patched_head = -4, 0, 0
avg_orig_diff = 0
avg_corrupt_diff = 0
total_num = 0
for inputs in tqdm(test_loader):
    input_ids : torch.LongTensor = inputs["input_ids"].to(device)
    input_ids = filter_inputs(input_ids, tokenizer)

    # con_input_ids = make_contrast_inputs(input_ids, test_dataset)
    con_input_ids = make_contrast_inputs_flip(input_ids, test_dataset)
    bz, seq_len = input_ids.size()

    orig_logits = hooked_model(input_ids, return_type="logits")
    
    hooked_model.reset_hooks()
    _, cache = hooked_model.run_with_cache(con_input_ids, return_type=None)
    
    pos_idx = (input_ids == tokenizer.eos_token_id).float().argmax(dim=-1) + 1 + patched_pos
    con_pos_idx = (con_input_ids == tokenizer.eos_token_id).float().argmax(dim=-1) + 1 + patched_pos

    fwd_hooks = []

    temp_hook0 = partial(replace_attn_result, head_idx=patched_head, pos_idx=pos_idx, con_pos_idx=con_pos_idx, corrupted_act=cache["blocks.0.attn.hook_result"])    #a0.0_qc
    fwd_hooks.append(("blocks.0.attn.hook_result", temp_hook0))

    # temp_hook1 = partial(replace_act, pos_idx=pos_idx, con_pos_idx=con_pos_idx, corrupted_act=cache["hook_embed"])
    # fwd_hooks.append(("hook_embed", temp_hook1))

    # temp_hook2 = partial(replace_attn_result, head_idx=patched_head, pos_idx=pos_idx+1, con_pos_idx=con_pos_idx+1, corrupted_act=cache["blocks.0.attn.hook_result"])    #a0.0_:
    # fwd_hooks.append(("blocks.0.attn.hook_result", temp_hook2))

    temp_hook3 = partial(replace_attn_result, head_idx=patched_head, pos_idx=pos_idx+1, con_pos_idx=con_pos_idx+1, corrupted_act=cache["blocks.1.attn.hook_result"])    #a1.0_:
    fwd_hooks.append(("blocks.1.attn.hook_result", temp_hook3))

    

    # temp_hook4 = partial(replace_act, pos_idx=pos_idx, con_pos_idx=con_pos_idx, corrupted_act=cache["hook_pos_embed"])
    # fwd_hooks.append(("hook_pos_embed", temp_hook4))

    # temp_hook5 = partial(replace_act, pos_idx=pos_idx+1, con_pos_idx=con_pos_idx+1, corrupted_act=cache["hook_pos_embed"])
    # fwd_hooks.append(("hook_pos_embed", temp_hook5))

    corrupt_logits = hooked_model.run_with_hooks(input_ids, 
                        return_type="logits",
                        fwd_hooks=fwd_hooks)

    pos_idx = (input_ids == tokenizer.eos_token_id).float().argmax(dim=-1) - 1
    con_pos_idx = (con_input_ids == tokenizer.eos_token_id).float().argmax(dim=-1) - 1
    arange_idx = torch.arange(bz, device=device, dtype=torch.long)
    orig_label = input_ids[arange_idx, pos_idx]
    con_label = con_input_ids[arange_idx, con_pos_idx]
    orig_diff = orig_logits[arange_idx, pos_idx-1, orig_label] - orig_logits[arange_idx, pos_idx-1, con_label]
    corrupt_diff = corrupt_logits[arange_idx, pos_idx-1, orig_label] - corrupt_logits[arange_idx, pos_idx-1, con_label]

    avg_orig_diff += orig_diff.sum().item()
    avg_corrupt_diff += corrupt_diff.sum().item()
    total_num += bz

avg_orig_diff /= total_num
avg_corrupt_diff /= total_num

# print("patching", "head 0.0: -3; head 1.0: -3")
print(avg_orig_diff - avg_corrupt_diff)
print(avg_orig_diff)
print(avg_corrupt_diff)


# patching head -4 0 0
# 22.123797705704128
# 0.04595471615669055



# patching head -3 1 0
# 22.113203947566106
# -15.820825889235277








# pos signal
# patching hook_pos_embed: -4, -3
# 22.123797705704128
# 6.996232523835011

# count signal
# patching head -4 0 0 + hook_embed: -4
# 22.123797705704128
# -4.669675784947322

# colon count signal
# patching head -3 0 0
# 22.118050869516225
# 16.031329928823617

# count signal + pos signal
# patching head -4 0 0 + hook_embed: -4 + hook_pos_embed: -4, -3
# 22.123797705704128
# -15.094680182667268


17.72978582419371
3.818436689176315
# = = = = = = = = = = = adversarial example = = = = = = 

# pos signal
# patching hook_pos_embed: -4, -3
# 18.218440536463007
# 18.218440536463007

# colon count signal
# patching head -3 0 0
# 18.21610897849135
# 18.06755975547726

# patching head -4 0 0  w/o hook_embed: -4
# 18.218024613484417
# -16.293337737417062


""" ===================== """
# patching pos_embed, -4, -3
# 15.127565181869116
# 22.123797705704128
# 6.996232523835011
  

# patching pos_embed: -4, -3; head 0.0: -4; embed: -4
# 37.218477888371396
# 22.123797705704128
# -15.094680182667268

            # patching pos_embed: -4, -3; head 0.0: -4
            # 22.483026667610805
            # 22.123797705704128
            # -0.35922896190667764
            

# patching pos_embed: -4, -3; head 0.0: -4, -3; embed: -4
# 41.182994137807995
# 22.123797705704128
# -19.059196432103867

            # patching pos_embed: -4, -3; head 0.0: -4, -3
            # 22.729386761871975
            # 22.123797705704128
            # -0.6055890561678471

# patching pos_embed: -4, -3; head 0.0: -4, -3; embed: -4; head 1.0: -3
# 44.25053480756711
# 22.123797705704128
# -22.126737101862982


# reverse =============

# patching head 1.0: -3
# 37.941413723207134
# 22.123797705704128
# -15.817616017503004

# patching head 1.0: -3; head 0.0: -3
# 42.94260998848157
# 22.123797705704128
# -20.818812282777444

# patching head 1.0: -3; head 0.0: -3; head 0.0: -4, embed: -4; 
# 42.94260998848157
# 22.123797705704128
# -20.818812282777444

# patching head 1.0: -3; head 0.0: -3; head 0.0: -4, embed: -4; pos_embed: -4, -3
# 44.25053480756711
# 22.123797705704128
# -22.126737101862982

{"forward": [15.127565181869116, 37.218477888371396, 41.182994137807995, 44.25053480756711], \
 "backward": [37.941413723207134, 42.94260998848157, 42.94260998848157, 44.25053480756711]}

""" ===================== adv ================== """

# patching pos_embed, -4, -3
# 0.0
# 18.221216658529357
# 18.221216658529357

# patching pos_embed, -4, -3; head 0.0: -4, embed: -4;
# 34.51589613361831
# 18.227600409641404
# -16.28829572397691

# patching pos_embed, -4, -3; head 0.0: -4, embed: -4; head 0.0: -3
# 34.55062738356659
# 18.223079665958085
# -16.327547717608503

# patching pos_embed, -4, -3; head 0.0: -4, embed: -4; head 0.0: -3; head 1.0: -3
# 35.5305911667604
# 18.222514747469248
# -17.308076419291154


# inverse =========
# head 1.0: -3
# 35.4282617410221
# 18.226286422327192
# -17.20197531869491

# head 0.0: -3; head 1.0: -3
# 35.5377205708684
# 18.218068962456744
# -17.31965160841166

# head 0.0: -4, embed: -4; head 0.0: -3; head 1.0: -3
# 35.532123512610454
# 18.220953584736804
# -17.311169927873653

# pos_embed, -4, -3; head 0.0: -4, embed: -4; head 0.0: -3; head 1.0: -3
# 35.53392705296929
# 18.21664127977963
# -17.31728577318966


{"forward": [0.0, 34.51589613361831, 34.55062738356659, 35.5305911667604], \
 "backward": [35.4282617410221, 35.5377205708684, 35.532123512610454, 35.53392705296929]}

# additional result:
# patching head 0.0: -3     # a0.0_:
# 0.14948562072094518
# 18.22876972529598
# 18.079284104575034

#  a1.0_: and a0.0_qc
# 35.431088127224
# 18.2261600181203
# -17.204928109103697