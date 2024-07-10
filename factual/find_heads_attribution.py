from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
from transformer_lens import HookedTransformer
import re
import json
from tqdm import tqdm
import random

torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", add_bos_token=True)

with open("../../datasets/factual/known_1000.json", "r") as f:
    examples = json.load(f)
text_inputs = [item["prompt"] for item in examples]
print(text_inputs)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
probed_model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
probed_model.resize_token_embeddings(probed_model.config.vocab_size+1)
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

hooked_model.eval()
del probed_model
hooked_model.set_use_attn_result(True)

def hook_site(name: str):
    if name.endswith("hook_result") or name.endswith("hook_resid_mid") or name.endswith("hook_resid_pre"):
        block_idx = int(re.search(r"blocks\.(\d+)\.", name).group(1))
        if block_idx >= 24:
            return True
    return False

cache = hooked_model.add_caching_hooks(hook_site)

bz = 32
thr = 0.02
heads_active_freq = {f"{i}.{j}": 0 for i in range(24, hooked_model.cfg.n_layers) for j in range(hooked_model.cfg.n_heads)}
for i in tqdm(range(0, len(text_inputs), bz)):
    text_batch = text_inputs[i:i+bz]
    tokens_batch = hooked_model.to_tokens(text_batch)
    hooked_model(tokens_batch, return_type=None)

    seq_lengths = torch.eq(tokens_batch, tokenizer.pad_token_id).int().argmax(-1) - 1
    arange_idx = torch.arange(tokens_batch.size(0), device=device)

    for j in range(24, hooked_model.cfg.n_layers):
        whole_vec = cache[f"blocks.{j}.hook_resid_mid"][arange_idx, seq_lengths].contiguous()   # [batch, d_model]
        parts_vec = cache[f"blocks.{j}.attn.hook_result"][arange_idx, seq_lengths].transpose(0, 1)   # [batch, pos, head_index, d_model] -> [head_idx, batch, d_model]
        parts_vec = torch.cat([cache[f"blocks.{j}.hook_resid_pre"][arange_idx, seq_lengths].unsqueeze(0), parts_vec], dim=0)

        temp_whole_vec = whole_vec.unsqueeze(0).expand_as(parts_vec)
        distance = torch.nn.functional.pairwise_distance(parts_vec, temp_whole_vec, p=1)

        whole_norm = torch.norm(whole_vec, p=1, dim=-1)
        proximity = (whole_norm.unsqueeze(0) - distance).clip(min=1e-5)

        proximity /= proximity.sum(dim=0, keepdim=True) # head_idx, batch

        proximity = proximity[1:].contiguous()
        # print("==="*10)
        # print(j)
        # print(proximity[:, 0]>thr)

        temp_freq = (proximity > thr).sum(dim=1)
        for k, n in enumerate(temp_freq.tolist()):
            heads_active_freq[f"{j}.{k}"] += n

print(heads_active_freq)

for k, v in sorted(list(heads_active_freq.items()), key=lambda x: x[1]):
    print(k, "\t", v)


# 31.0 	 30
# 33.0 	 32
# 38.22 	 34
# 29.9 	 38
# 37.7 	 38
# 32.12 	 42
# 31.8 	 49
# 24.24 	 50
# 28.3 	 51
# 25.7 	 52
# 28.21 	 53
# 27.16 	 56
# 42.24 	 66
# 31.4 	 75
# 34.20 	 76
# 30.23 	 77
# 24.8 	 79
# 30.8 	 80
# 25.10 	 90
# 33.9 	 98
# 32.15 	 119
# 30.1 	 120
# 29.20 	 161
# 36.17 	 170
# 35.19 	 486