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

from train import *

random.seed(0)
torch.set_grad_enabled(False)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = "../training_outputs/addition_fixed/checkpoint-59350"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    model = model.to(device)

    tokenizer = customTokenizer()

    # dataset, _ = make_dataset(tokenizer, train_ratio=1.0)
    _, dataset = make_dataset(tokenizer)
    
    data_collator = customCollator(tokenizer.pad_token_id)

    test_loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=data_collator)

    correct_num = 0

    total_9_num = 0
    correct_9_num = 0
    for inputs in test_loader:
        input_ids : torch.LongTensor = inputs["input_ids"].to(device)
        bz, seq_len = input_ids.size()
        equal_pos = (input_ids == tokenizer.vocab["="]).float().argmax(dim=-1)
        after_equal = torch.arange(seq_len, device=device).unsqueeze(0).expand(bz, -1) > equal_pos.unsqueeze(1)
        generated_ids = greedy_generation(model, input_ids.masked_fill(after_equal, tokenizer.pad_token_id), tokenizer)
        correct_num += (generated_ids == input_ids).all(dim=-1).sum().item()

        # check if all wrong are 9
        # wrong_mask = ~(generated_ids == input_ids).all(dim=-1)
        # if wrong_mask.sum() > 0:
            
        #     special_m = (input_ids[wrong_mask][:, 2] + input_ids[wrong_mask][:, 6] != 9)
        #     try:
        #         assert special_m.sum() == 0
        #     except:
        #         print("input_ids", input_ids[wrong_mask])
        #         print("pred", generated_ids[wrong_mask])

        # check acc when tens sum == 9
        mask = (input_ids[:, 2] + input_ids[:, 6]) == 9
        if mask.sum().item() == 0:
            continue
        total_9_num += mask.sum().item()
        correct_9_num += (generated_ids[mask] == input_ids[mask]).all(dim=-1).sum().item()

        # mask1 = (input_ids[:, 2] + input_ids[:, 6] == 9) & (input_ids[:, -1] == tokenizer.pad_token_id)
        # total_3digit_9_num += mask1.sum().item()
        # correct_3digit_9_num += (generated_ids[mask1] == input_ids[mask1]).all(dim=-1).sum().item()

        # mask2 = (input_ids[:, 2] + input_ids[:, 6] == 9) & (input_ids[:, -1] == tokenizer.eos_token_id)
        # total_4digit_9_num += mask2.sum().item()
        # correct_4digit_9_num += (generated_ids[mask2] == input_ids[mask2]).all(dim=-1).sum().item()
        

    acc = correct_num / len(test_loader.dataset)
    print("accuracy", acc)

    print("9 acc")
    print(correct_9_num, total_9_num)
    print(correct_9_num / total_9_num)

    # print("3 digit")
    # print(correct_3digit_9_num, total_3digit_9_num)
    # print("accuracy for 9", correct_3digit_9_num / total_3digit_9_num)

    # print("4 digit")
    # print(correct_4digit_9_num, total_4digit_9_num)
    # print("accuracy for 9", correct_4digit_9_num / total_4digit_9_num)


# 65084 81000
# accuracy for 9 0.8035061728395062

# on train set
# accuracy 0.9804230452674897
# 9 acc: 48932 60825, 0.804471845458282
    
# on test set
# accuracy 0.9801333333333333
# 9 acc: 16152 20175, 0.8005947955390335

# 3 digit
# 29567 32400
# accuracy for 9 0.9125617283950618
# 4 digit
# 35517 48600
# accuracy for 9 0.7308024691358025