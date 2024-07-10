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
# torch.set_printoptions(sci_mode=False)

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

train_dataset, test_dataset = make_dataset(tokenizer)

data_collator = customCollator(tokenizer.pad_token_id)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=data_collator)



avg_orig_diff = 0
avg_corrupt_diff = 0

avg_act = 0
total_num = 0
for inputs in tqdm(test_loader):
    input_ids : torch.LongTensor = inputs["input_ids"].to(device)
    bz, seq_len = input_ids.size()

    _, cache = hooked_model.run_with_cache(input_ids, return_type=None)
    
    avg_act = avg_act + (cache["blocks.0.hook_resid_post"][:, 4].sum(dim=0))
    total_num += bz

avg_act /= total_num


avg_kl_div = 0
max_logit_diff = 0
max_logit_diff_ratio = 0
total_num = 0
for inputs in tqdm(test_loader):
    input_ids : torch.LongTensor = inputs["input_ids"].to(device)
    bz, seq_len = input_ids.size()

    orig_logits = hooked_model(input_ids, return_type="logits")
    orig_dist = F.log_softmax(orig_logits, dim=-1)

    max_logit_idx = orig_logits.argmax(dim=-1, keepdim=True)
    orig_max_logit = torch.gather(orig_logits, dim=2, index=max_logit_idx)
    assert (orig_max_logit == orig_logits.max(dim=-1, keepdim=True)[0]).all()

    fwd_hooks = []
    
    def replace_act(x, hook):
        x[:, 4] = avg_act.unsqueeze(0).expand(bz, -1)
        return x

    fwd_hooks.append(("blocks.0.hook_resid_post", replace_act))

    corrupt_logits = hooked_model.run_with_hooks(input_ids, 
                        return_type="logits",
                        fwd_hooks=fwd_hooks)
    corrupt_dist = F.log_softmax(corrupt_logits, dim=-1)

    kl_div = F.kl_div(corrupt_dist, orig_dist, log_target=True, reduction="none").sum(dim=-1).sum(dim=0)
    avg_kl_div = avg_kl_div + kl_div
    total_num += bz

    corrupt_max_logit = torch.gather(corrupt_logits, dim=2, index=max_logit_idx)

    max_logit_diff = max_logit_diff + (orig_max_logit - corrupt_max_logit).sum(dim=0)

    max_logit_diff_ratio = max_logit_diff_ratio + ((orig_max_logit - corrupt_max_logit) / orig_max_logit).sum(dim=0)



avg_kl_div /= total_num
max_logit_diff /= total_num
max_logit_diff_ratio /= total_num
print(avg_kl_div[8:12])
print(max_logit_diff[8:12])
print(max_logit_diff_ratio[8:12])


