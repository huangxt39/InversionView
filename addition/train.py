from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, GenerationConfig
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import math


# BOS_ID = 10
# PLUS_ID = 11
# EQUAL_ID = 12
# EOS_ID = 13
# PAD_ID = 14

MAX_LEN = 14
# VOCAB_LEN = 15


class customTokenizer():
    def __init__(self,):
        self.bos_token = "B"
        self.eos_token = "E"
        self.pad_token = "P"
        self.bos_token_id = 10
        self.eos_token_id = 13
        self.pad_token_id = 14
        self.special_token_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id]
        
        self.vocab = {str(i): i for i in range(10)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab["+"] = 11
        self.vocab["="] = 12

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        # only used by hooked_model.to_tokens()
        if type(strings) == str:
            strings = [strings]
        ids = []
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)



class Addition3Digit(Dataset):
    def __init__(self, data: list[tuple[int, int]], tokenizer: customTokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        a, b = self.data[idx]
        return self._number_to_ids(a, b)

    def _number_to_ids(self, a, b):
        s = str(a) + "+" + str(b) + "=" + str(a+b)
        s = self.tokenizer.bos_token + s + self.tokenizer.eos_token
        s += self.tokenizer.pad_token * (MAX_LEN-len(s))
        return list(map(lambda x: self.tokenizer.vocab[x], s))

    def __len__(self):
        return len(self.data)
        
        
def make_dataset(tokenizer, train_ratio=0.75):
    random.seed(0)

    data_points = [(i, j) for i in range(100, 1000) for j in range(100, 1000)]
    random.shuffle(data_points)
    train_num = int(train_ratio * len(data_points))
    train_dataset = Addition3Digit(data_points[:train_num].copy(), tokenizer)
    test_dataset = Addition3Digit(data_points[train_num:].copy(), tokenizer)
    return train_dataset, test_dataset


class customCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        input_ids = torch.LongTensor(examples)

        batch = {"input_ids": input_ids}

        labels = input_ids.clone()
        labels[labels == self.pad_id] = -100
        batch["labels"] = labels
        return batch

@torch.no_grad()
def greedy_generation(model: GPT2LMHeadModel, prompts: torch.LongTensor, tokenizer: customTokenizer):
    generated_ids = []
    bz, seq_len = prompts.size()
    device = prompts.device

    inputs = {
        "input_ids": prompts[:, 0:1],
        "use_cache": True,
    }

    generated_ids = inputs["input_ids"]
    terminated = torch.zeros(bz, dtype=torch.bool, device=device)
    for i in range(1, seq_len):
        outputs = model(**inputs)
        logits, past_kv = outputs.logits, outputs.past_key_values
        next_ids = logits.argmax(dim=-1)
        next_ids.masked_fill_(terminated.unsqueeze(1), tokenizer.pad_token_id)
        next_ids = torch.where(prompts[:, i:i+1] == tokenizer.pad_token_id, next_ids, prompts[:, i:i+1])

        inputs = {
            "input_ids": next_ids,
            "past_key_values": past_kv,
            "use_cache": True
        }

        generated_ids = torch.hstack([generated_ids, next_ids])

        terminated = terminated | (next_ids.squeeze(1) == tokenizer.eos_token_id)

    return generated_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--arch_d", type=int, default=32)
    parser.add_argument("--arch_l", type=int, default=2)
    parser.add_argument("--arch_h", type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = customTokenizer()
    train_dataset, test_dataset = make_dataset(tokenizer)

    print("train set len", len(train_dataset))
    print("test set len", len(test_dataset))

    cfg = GPT2Config(vocab_size=len(tokenizer), 
                n_positions=MAX_LEN,  # for eos
                n_embd=args.arch_d,
                n_layer=args.arch_l,
                n_head=args.arch_h,
                bos_token_id=tokenizer.bos_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                attn_pdrop=0,
                )
    model = GPT2LMHeadModel(cfg)

    training_args = TrainingArguments(
        output_dir="../training_outputs/addition_fixed",    # 0.9801333333333333
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="no",
        num_train_epochs=args.num_epoch,
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        report_to="none",
    )

    data_collator = customCollator(tokenizer.pad_token_id)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    class ComputeAccuracy(TrainerCallback):
        def on_epoch_end(self, train_args: TrainingArguments, state: TrainerState, control: TrainerControl, model: GPT2LMHeadModel, **kwargs):
            if round(state.epoch) %2 == 0: # 15
                model.eval()
                with torch.no_grad():
                    correct_num = 0
                    for inputs in test_loader:
                        input_ids : torch.LongTensor = inputs["input_ids"].to(device)
                        bz, seq_len = input_ids.size()
                        equal_pos = (input_ids == tokenizer.vocab["="]).float().argmax(dim=-1)
                        after_equal = torch.arange(seq_len, device=device).unsqueeze(0).expand(bz, -1) > equal_pos.unsqueeze(1)
                        generated_ids = greedy_generation(model, input_ids.masked_fill(after_equal, tokenizer.pad_token_id), tokenizer)
                        correct_num += (generated_ids == input_ids).all(dim=-1).sum().item()
                    acc = correct_num / len(test_loader.dataset)
                    print("accuracy", acc)
                model.train()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[ComputeAccuracy],
    )

    trainer.train()