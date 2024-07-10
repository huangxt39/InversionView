from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import torch
import nltk
import re
import itertools
from tqdm import tqdm

torch.set_grad_enabled(False)


dataset = load_dataset("JeanKaddour/minipile")
subset = dataset["train"].shuffle(seed=0)
subset = subset.select(range(len(subset)//10))

def split_sents(examples):
    sents = []
    for text in examples["text"]:
        sents.extend( list(filter(lambda x: len(x) <= 100, nltk.sent_tokenize(text))) )
    return {"sentence": sents}

print(subset)
subset = subset.map(split_sents, batched=True, remove_columns=subset.column_names, num_proc=4)
print(subset)


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

try:
    for i in tqdm(range(0, len(subset), bz)):
        text_batch = subset.select(range(i, min(i+bz, len(subset))))["sentence"]
        tokens_batch = hooked_model.to_tokens(text_batch)
        assert (tokens_batch[:, 0] == tokenizer.bos_token_id).all()
        hooked_model(tokens_batch, return_type=None)

        pad_mask = tokens_batch == tokenizer.pad_token_id

        for h in text_per_head:
            block_idx, head_idx = h.split(".")
            block_idx, head_idx = int(block_idx), int(head_idx)
            # [batch, head_index, query_pos, key_pos]
            attn_on_bos = cache[f"blocks.{block_idx}.attn.hook_pattern"][:, head_idx, :, 0].contiguous()
            attn_on_bos.masked_fill_(pad_mask, 1.0)
            mask = (attn_on_bos.min(dim=1)[0] < 0.6).tolist()

            text_per_head[h].extend( [t for t, m in zip(text_batch, mask) if m] )

except:
    print("WARNING: error raised")

text_per_head_last_pos = load_from_disk("../datasets/factual/data_last_pos")

for h in text_per_head:
    text_per_head[h].extend( list(map(lambda x: x[1], text_per_head_last_pos[h]["sentence"])) )

text_per_head = DatasetDict( {h+"_all": Dataset.from_dict({"sentence": t}) for h, t in text_per_head.items()} )
text_per_head = text_per_head.shuffle(seed=0)
print(text_per_head)
 

all_data = {k: v for k, v in itertools.chain(text_per_head_last_pos.items(), text_per_head.items())}
all_data = DatasetDict(all_data)
print('all data')
print(all_data)
all_data.save_to_disk("../datasets/factual/data_all")