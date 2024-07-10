from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, load_from_disk
import json
import os
import pandas as pd
from tqdm import tqdm
import re
from glob import glob
import random
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from transformer_lens import HookedTransformer


torch.set_grad_enabled(False)


# make dataset from counterfact
# https://rome.baulab.info/data/dsets/
# download counterfact.json and known_1000.json

def clean_prefix(text):
    return re.split(r'\. |Category\:\S+ |\.["\)] |! ', text)[-1]

data_root_path = "../datasets/factual"
with open(os.path.join(data_root_path, "counterfact.json"), "r") as f:
    orig_dataset = json.load(f)

text = []
for i, item in tqdm(enumerate(orig_dataset)):
    str1 = item["requested_rewrite"]["prompt"]
    str1 = str1.replace(r"{}", item["requested_rewrite"]["subject"])

    rephrased_str = list(map(clean_prefix, item["paraphrase_prompts"]))

    prompts_true = [str1,] + rephrased_str + item["neighborhood_prompts"]
    prompts_true = [(p, p+" "+item["requested_rewrite"]["target_true"]["str"]) for p in prompts_true]
    prompts_new = item["attribute_prompts"]
    prompts_new = [(p, p+" "+item["requested_rewrite"]["target_new"]["str"]) for p in prompts_new]
    text.extend(prompts_true)
    text.extend(prompts_new)


# make dataset from BEAR
# https://github.com/lm-pub-quiz/BEAR

bear_path = "../datasets/factual/BEAR-big"
def make_sents(templates: list[str], subjs: list[str], obj: str):
    sents = []
    for template in templates:
        for subj in subjs:
            temp_template = template[: template.index("[Y]")].rstrip()
            sents.append( (temp_template.replace("[X]", subj), 
                           template.replace("[X]", subj).replace("[Y]", obj)) )
            # sents.append( template.replace("[X]", subj).replace("[Y]", obj).rstrip(".") )
    return sents


with open(os.path.join(bear_path, "metadata_relations.json"), "r") as f:
    metadata = json.load(f)
del metadata["P414"]

for relation_k, relation_v in tqdm(metadata.items()):
    templates = relation_v["templates"]
    templates = list(filter(lambda x: x.index("[X]") < x.index("[Y]"), templates))
    # templates = list(map(lambda x: x[: x.index("[Y]")].rstrip(), templates))

    with open(os.path.join(bear_path, relation_k+".jsonl"), "r") as f:
        for line in f.readlines():
            instance = json.loads(line)
            subjs = [instance["sub_label"],] + list(filter(lambda x: x.isascii(), instance["sub_aliases"]))
            obj = instance["obj_label"]

            text.extend(make_sents(templates, subjs, obj))

print("len of text", len(text))
print(text[:3])
print(text[-3:])
# select text for each head

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", add_bos_token=True)
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

def hook_site(name: str):
    if name.endswith("hook_pattern"):
        block_idx = int(re.search(r"blocks\.(\d+)\.", name).group(1))
        if block_idx >= 24:
            return True
    return False

cache = hooked_model.add_caching_hooks(hook_site)


bz = 64
selected_heads = '31.0 33.0 38.22 29.9 37.7 32.12 31.8 24.24 28.3 25.7 28.21 27.16 42.24 31.4 34.20 30.23 24.8 30.8 25.10 33.9 32.15 30.1 29.20 36.17 35.19'
text_per_head = {h: [] for h in selected_heads.split(" ")}

for i in tqdm(range(0, len(text), bz)):
    text_tuple = text[i:i+bz]
    text_batch = list(map(lambda x: x[0], text_tuple))
    tokens_batch = hooked_model.to_tokens(text_batch)
    hooked_model(tokens_batch, return_type=None)

    seq_lengths = torch.eq(tokens_batch, tokenizer.pad_token_id).int().argmax(-1) - 1
    arange_idx = torch.arange(tokens_batch.size(0), device=device)

    for h in text_per_head:
        block_idx, head_idx = h.split(".")
        block_idx, head_idx = int(block_idx), int(head_idx)
        # [batch, head_index, query_pos, key_pos]
        attn_on_bos = cache[f"blocks.{block_idx}.attn.hook_pattern"][:, head_idx][arange_idx, seq_lengths, 0]
        mask = (attn_on_bos < 0.6).tolist()

        text_per_head[h].extend( [t for t, m in zip(text_tuple, mask) if m] )

text_per_head = DatasetDict( {h: Dataset.from_dict({"sentence": t}) for h, t in text_per_head.items()} )
text_per_head = text_per_head.shuffle(seed=0)
print(text_per_head)
text_per_head.save_to_disk("../datasets/factual/data_last_pos")