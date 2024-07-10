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
import string
from itertools import combinations


# BOS_ID = 10
# PLUS_ID = 11
# EQUAL_ID = 12
# EOS_ID = 13
# PAD_ID = 14

MAX_LEN = 9 * 3 + 6
# VOCAB_LEN = 15



class customTokenizer():
    def __init__(self,):
        self.bos_token = "B"
        self.eos_token = "E"
        self.pad_token = "P"
        all_c = string.ascii_lowercase + "123456789|:" + self.bos_token + self.eos_token + self.pad_token
        self.vocab = {c: i for i, c in enumerate(all_c)}
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.special_token_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id]

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


class CharacterNum(Dataset):
    def __init__(self, data: list[tuple[int, int]], tokenizer: customTokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        string = self.data[idx]
        return self._number_to_ids(string)

    def _number_to_ids(self, s):
        s = self.tokenizer.bos_token + s + self.tokenizer.eos_token
        s += self.tokenizer.pad_token * (MAX_LEN-len(s))
        return list(map(lambda x: self.tokenizer.vocab[x], s))

    def __len__(self):
        return len(self.data)
        
def get_new_config(existing):
    num_a = random.randint(1, 9)
    num_b = random.randint(1, 9)
    num_c = random.randint(1, 9)
    while (num_a, num_b, num_c) in existing:
        num_a = random.randint(1, 9)
        num_b = random.randint(1, 9)
        num_c = random.randint(1, 9)
    return (num_a, num_b, num_c)
        
def make_dataset(tokenizer, train_ratio=0.75):
    random.seed(0)
    data_points = []
    for a, b, c in combinations(string.ascii_lowercase, 3):
        existing = []
        for i in range(200):
            num_a, num_b, num_c = get_new_config(existing)
            existing.append((num_a, num_b, num_c))
            char_list = [a] * num_a + [b] * num_b + [c] * num_c
            
            random.shuffle(char_list)
            data_point = "".join(char_list) + "|" + a + ":" + str(num_a)
            data_points.append(data_point)

            random.shuffle(char_list)
            data_point = "".join(char_list) + "|" + b + ":" + str(num_b)
            data_points.append(data_point)

            random.shuffle(char_list)
            data_point = "".join(char_list) + "|" + c + ":" + str(num_c)
            data_points.append(data_point)


    random.shuffle(data_points)
    train_num = int(train_ratio * len(data_points))
    train_dataset = CharacterNum(data_points[:train_num].copy(), tokenizer)
    test_dataset = CharacterNum(data_points[train_num:].copy(), tokenizer)
    return train_dataset, test_dataset


class customCollator():
    def __init__(self, special_id):
        self.special_id = special_id

    def __call__(self, examples):
        input_ids = torch.LongTensor(examples)

        batch = {"input_ids": input_ids}

        labels = input_ids.clone()
        mask = labels != self.special_id
        mask = torch.cat([torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device), mask[:, :-1]], dim=1)
        labels[mask] = -100
        batch["labels"] = labels
        return batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--arch_d", type=int, default=64)
    parser.add_argument("--arch_l", type=int, default=2)
    parser.add_argument("--arch_h", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(1)
    print(args)
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
                resid_pdrop=0,
                embd_pdrop=0,
                )
    model = GPT2LMHeadModel(cfg)

    training_args = TrainingArguments(
        output_dir="../training_outputs/counting",   # 0.9953205128205128
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

    data_collator = customCollator(tokenizer.vocab[":"])

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    class ComputeAccuracy(TrainerCallback):
        def on_epoch_end(self, train_args: TrainingArguments, state: TrainerState, control: TrainerControl, model: GPT2LMHeadModel, **kwargs):
            if round(state.epoch) % 2 == 0: # 15
                model.eval()
                with torch.no_grad():
                    correct_num = 0
                    for inputs in test_loader:
                        input_ids : torch.LongTensor = inputs["input_ids"].to(device)
                        bz, seq_len = input_ids.size()
                        labels = inputs["labels"].to(device)
                        logits = model(input_ids=input_ids).logits
                        pred = logits.argmax(dim=-1)
                        pred = pred[input_ids==tokenizer.vocab[":"]].view(bz)
                        labels = input_ids[labels!=-100].view(bz)

                        # for i in range(3):
                        #     print("".join(tokenizer.convert_ids_to_tokens(input_ids[i].tolist())))
                        #     print("".join(tokenizer.convert_ids_to_tokens(pred[i].tolist())))
                        #     print("".join(tokenizer.convert_ids_to_tokens(labels[i].tolist())))
                        # exit()
                        correct_num += (pred == labels).sum().item()
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