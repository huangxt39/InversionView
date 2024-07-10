from ioi_dataset import IOIDataset, NAMES
from transformers import AutoTokenizer
import torch
import pickle
import random
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")

ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=250000,
        seed = 0,
    )


def merge_prompts_and_pos(prompts: list[dict[str, str]], word_idx: dict[str, torch.LongTensor]):
    prompts = prompts.copy()
    assert len(prompts) ==  len(word_idx["IO"])
    for i in tqdm(range(len(prompts))):
        for k in word_idx.keys():
            prompts[i][f"pos_{k}"] = word_idx[k][i].item()
    return prompts

    
merged_prompts = []
merged_prompts.extend(merge_prompts_and_pos(ioi_dataset.ioi_prompts, ioi_dataset.word_idx))
random.seed(1)
random.shuffle(merged_prompts)

for p in merged_prompts:
    assert type(p["text"]) == str, p

with open("../datasets/ioi_prompts.pkl", "wb") as f:
    pickle.dump(merged_prompts, f)
print("data saved!")