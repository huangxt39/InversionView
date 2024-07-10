import torch
import torch.nn as nn
import torch.nn.functional as F
import circuitsvis as cv

import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import os
import random
import json

from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, HookedTransformerConfig
from train import*

torch.set_grad_enabled(False)
torch.set_printoptions(sci_mode=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

probed_model_path = "../training_outputs/counting/checkpoint-914100"
model = GPT2LMHeadModel.from_pretrained(probed_model_path).to(device)
tokenizer = customTokenizer()

train_dataset, test_dataset = make_dataset(tokenizer)

data_collator = customCollator(tokenizer.vocab[":"])

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=data_collator)

model.eval()
                
correct_num = 0
for inputs in test_loader:
    input_ids : torch.LongTensor = inputs["input_ids"].to(device)
    bz, seq_len = input_ids.size()
    labels = inputs["labels"].to(device)
    logits = model(input_ids=input_ids).logits
    pred = logits.argmax(dim=-1)
    pred = pred[input_ids==tokenizer.vocab[":"]].view(bz)
    labels = input_ids[labels!=-100].view(bz)

    correct_num += (pred == labels).sum().item()
acc = correct_num / len(test_loader.dataset)
print("accuracy", acc)
