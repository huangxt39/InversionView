from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import re
import json
from tqdm import tqdm
import random
import pickle
from ioi_dataset import NAMES

torch.set_grad_enabled(False)
random.seed(0)

hooked_model = HookedTransformer.from_pretrained("gpt2")
tokenizer = hooked_model.tokenizer
hooked_model.set_use_attn_result(True)

layer_idx = 7
head_idx = 9
sel_pos = "pos_end"     # (obj["pos_S+1"]+1, obj["pos_S2"]+1, obj["pos_end"]+1) +1 because of BOS
cache = hooked_model.add_caching_hooks(f"blocks.{layer_idx}.attn.hook_result")

# qualitative
example_obj = {"text": "When Mary and John went to the store, John gave a drink to",
                   "pos_S+1": 4,
                   "pos_S2": 9,
                   "pos_end": 13}
tokens = hooked_model.to_tokens(example_obj["text"])    # with batch dim, with BOS
hooked_model(tokens, stop_at_layer=layer_idx+1)
head_output = cache[f"blocks.{layer_idx}.attn.hook_result"][:, example_obj[sel_pos]+1, head_idx].clone() # [batch, pos, head_index, d_model]
# head_output = F.layer_norm(head_output, [head_output.size(-1),])
logit_effect = torch.matmul(head_output, hooked_model.unembed.W_U)[0]

_, max_idx = logit_effect.topk(k=15)
# top_tokens = tokenizer.convert_ids_to_tokens()
print(", ".join(map(lambda x: tokenizer.decode(x), max_idx.tolist())))
print()

_, min_idx = logit_effect.topk(k=15, largest=False)
print(", ".join(map(lambda x: tokenizer.decode(x), min_idx.tolist())))


name_list = NAMES.copy()
name_list.extend(["Anna", "Nikki", "Cindy", "Fiona", "Brenda", "Audrey", "Marina", "Natalie", "Ryder", "Jasper", "Irwin", "Jake", "Bobby", "Bruno"])

all_names = ["Ġ"+n for n in name_list] + name_list
all_names = list(filter(lambda n: tokenizer.convert_tokens_to_ids(n) != tokenizer.unk_token_id, all_names))
all_name_ids = tokenizer.convert_tokens_to_ids(all_names)
print(len(all_name_ids))
assert tokenizer.unk_token_id not in all_name_ids

data_path = "../../datasets/ioi_prompts.pkl"
with open(data_path, "rb") as f:
    rollouts: list[dict[str, str]] = pickle.load(f)
# {'[PLACE]': 'hospital', '[OBJECT]': 'basketball', 'text': 'The local big hospital Jacob and Jason went to had a basketball. Jason gave it to Jacob', 'IO': 'Jacob', 'S': 'Jason', 'TEMPLATE_IDX': 11, 'pos_IO': 4, 'pos_IO-1': 3, 'pos_IO+1': 5, 'pos_S': 6, 'pos_S-1': 5, 'pos_S+1': 7, 'pos_S2': 13, 'pos_end': 16, 'pos_starts': 0, 'pos_punct': 12}
def check_if_ioi(obj):
    tokens = obj["text"].split(" ")
    return (tokens.count(obj["IO"]) == 2) and (tokens.count(obj["S"]) == 2)

rollouts = list(filter(check_if_ioi, rollouts))
random_idx = random.sample(list(range(len(rollouts))), 1000)


sName_top = 0
sName_top_first = 0
sName_bottom = 0
sName_bottom_first = 0

ioName_top = 0
ioName_top_first = 0
ioName_bottom = 0
ioName_bottom_first = 0

def whether_inside(ref_tokens: list[str], target_tokens: list[str]):
    for token in target_tokens:
        if token in ref_tokens:
            return True
    return False

for i in tqdm(random_idx):
    obj = rollouts[i]
    tokens = hooked_model.to_tokens(obj["text"])    # with batch dim, with BOS
    hooked_model(tokens, stop_at_layer=layer_idx+1)
    head_output = cache[f"blocks.{layer_idx}.attn.hook_result"][:, obj[sel_pos]+1, head_idx].clone() # [batch, pos, head_index, d_model]
    # head_output = F.layer_norm(head_output, [head_output.size(-1),])
    logit_effect = torch.matmul(head_output, hooked_model.unembed.W_U)[0]

    # print(obj["text"], tokenizer.convert_ids_to_tokens(tokens[0, obj[sel_pos]+1].item()))
    
    max_v, max_idx = logit_effect.topk(k=30)
    top_tokens = tokenizer.convert_ids_to_tokens(max_idx.tolist())

    min_v, min_idx = logit_effect.topk(k=30, largest=False)
    bottom_tokens = tokenizer.convert_ids_to_tokens(min_idx.tolist())
    

    s_name = obj["S"]
    assert ("Ġ"+s_name) in all_names
    s_name = ["Ġ"+s_name] if s_name not in all_names else [s_name, "Ġ"+s_name]

    if whether_inside(top_tokens, s_name):
        sName_top += 1
        for t in top_tokens:
            if t in all_names:
                if t in s_name:
                    sName_top_first += 1
                break

    if whether_inside(bottom_tokens, s_name):
        sName_bottom += 1
        for t in bottom_tokens:
            if t in all_names:
                if t in s_name:
                    sName_bottom_first += 1
                break

    io_name = obj["IO"]
    assert ("Ġ"+io_name) in all_names
    io_name = ["Ġ"+io_name] if io_name not in all_names else [io_name, "Ġ"+io_name]

    if whether_inside(top_tokens, io_name):
        ioName_top += 1
        for t in top_tokens:
            if t in all_names:
                if t in io_name:
                    ioName_top_first += 1
                break
        

    if whether_inside(bottom_tokens, io_name):
        ioName_bottom += 1
        for t in bottom_tokens:
            if t in all_names:
                if t in io_name:
                    ioName_bottom_first += 1
                break

print(sName_top, sName_top_first)
print(sName_bottom, sName_bottom_first)
print(ioName_top, ioName_top_first)
print(ioName_bottom, ioName_bottom_first)



            

