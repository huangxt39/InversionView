from datasets import load_dataset
import json
from tqdm import tqdm
from transformers import AutoTokenizer

dataset = load_dataset("JeanKaddour/minipile")
subset = dataset["train"]

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", add_bos_token=True)
token_freq = [0 for _ in range(len(tokenizer))]
for i in tqdm(range(len(subset))):
    input_ids = tokenizer(subset[i]["text"])["input_ids"]
    for idx in input_ids:
        token_freq[idx] += 1
print(token_freq[tokenizer.vocab["Ġthe"]])
print(token_freq[tokenizer.vocab["Ġcurrent"]])
print(token_freq[tokenizer.vocab["ĠApple"]])
with open("token_freq.json", "w") as f:
    json.dump(token_freq, f)