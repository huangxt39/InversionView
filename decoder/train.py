import math
import argparse
from tqdm import tqdm
import random
import os
import pickle
import re
import uuid 
from typing import Optional
from datasets import load_dataset, load_from_disk
from collections import OrderedDict
import threading
import queue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizerFast, TrainerCallback, TrainingArguments, logging
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainerCallback, TrainerControl, TrainerState, GenerationConfig
from transformers.generation.utils import GenerationMode

from transformer_lens import HookedTransformer

from utils import *
from model import CustomGPT2LMHeadModel
from generate import get_test_activation, generate_and_print, run_test

logging.set_verbosity_info()

def choose_random_pos(
        args,
        tokens_batch: torch.LongTensor,
        tokenizer: PreTrainedTokenizerFast,
    ):
    batch_size, seq_len = tokens_batch.size()
    device = tokens_batch.device
    pad_mask = (tokens_batch == tokenizer.pad_token_id) | (tokens_batch == tokenizer.eos_token_id) | (tokens_batch == tokenizer.bos_token_id)
    arange_idx = torch.arange(batch_size, dtype=torch.long, device=device)

    # remove last few tokens
    if args.probed_task == "counting":
        pos_temp = (tokens_batch == tokenizer.vocab["|"]).float().argmax(dim=1, keepdim=True)
        arange_temp = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        mask_temp = (arange_temp <= pos_temp) | (arange_temp > pos_temp+2)
        pad_mask = pad_mask | mask_temp

    random_temp = torch.rand_like(tokens_batch, dtype=torch.float)
    random_temp.masked_fill_(pad_mask, -1e6)
    if args.pos_bias is not None:  
        bias = torch.linspace(0, args.pos_bias, seq_len, device=device).unsqueeze(0)
        random_temp += bias

    sel_pos = random_temp.argmax(dim=1)
    return arange_idx, sel_pos

def init_hooked_model(args, device: torch.device, probed_model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast,):
    hooked_model = HookedTransformer.from_pretrained(
            "gpt2",
            hf_model=probed_model,
            device=device,
            tokenizer=tokenizer if isinstance(tokenizer, PreTrainedTokenizerFast) else None,
            n_embd=probed_model.config.n_embd,
            n_layer=probed_model.config.n_layer,
            n_head=probed_model.config.n_head,
            vocab_size=probed_model.config.vocab_size,
            n_ctx=probed_model.config.n_positions,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        hooked_model.tokenizer = tokenizer
    elif args.probed_task == "fact":
        hooked_model.cfg.n_ctx = args.max_len
    print("init hooked_model at", hooked_model.embed.W_E.device)
    hooked_model.eval()
    return hooked_model



def collect_data_thread(
        args,
        device: torch.device,
        read_lock: threading.Lock,
        write_q: queue.Queue,
        probed_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizerFast,
        act_dataset: ActivationDataset, 
        rollouts: rolloutManagerIOI,
        probed_acts: list[str], 
        seed: Optional[int] = None,
        prob_weight: Optional[dict[str, float]] = None
    ):
    """ init hooked model on this device """
    hooked_model = init_hooked_model(args, device, probed_model, tokenizer)
    if seed is not None:
        torch.manual_seed(seed)
    """ collect activation and corresponding rollout"""
    
    batch_size = args.caching_batch_size

    if prob_weight is None:
        prob_weight = OrderedDict({k: 0.0 for k in probed_acts})
    random_choice = torch.distributions.Categorical(logits=torch.tensor(list(prob_weight.values()), device=device))

    with torch.no_grad():
        while len(act_dataset) < act_dataset.data_per_epoch:
            probed_act = list(prob_weight.keys())[random_choice.sample().item()]
            assert probed_act in probed_acts

            last_pos = False
            read_lock.acquire()
            if args.probed_task == "fact":
                last_pos = torch.rand(()).item() > 0.5
                text_batch = rollouts.next_batch(batch_size, probed_act, last_pos)
            else:
                text_batch = rollouts.next_batch(batch_size)
            read_lock.release()

            if last_pos:
                text_batch_short = list(map(lambda x: x[0] + tokenizer.eos_token, text_batch))  # without last label
                sel_pos = (hooked_model.to_tokens(text_batch_short) == tokenizer.eos_token_id).float().argmax(dim=1) - 1

                text_batch = list(map(lambda x: x[1], text_batch))  

            text_batch = list(map(lambda x: x + tokenizer.eos_token, text_batch))
            
            tokens_batch = hooked_model.to_tokens(text_batch)
            assert (tokens_batch[:,0] == tokenizer.bos_token_id).all()

            block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
            cache = add_necessary_hooks(hooked_model, [probed_act])
            hooked_model(tokens_batch, stop_at_layer=block_idx+1)
            activation = retrieve_act(probed_act, cache)
            end_caching(hooked_model)

            if args.probed_task == "fact":
                arange_idx = torch.arange(tokens_batch.size(0), dtype=torch.long, device=device)
                if not last_pos:
                    head_idx = int(re.search(r"hook_result\.(\d+)", probed_act).group(1))
                    weights_on_bos = cache[f"blocks.{block_idx}.attn.hook_pattern"][:, head_idx, :, 0].clone()
                    weights_on_bos.masked_fill_((tokens_batch == tokenizer.pad_token_id) | (tokens_batch == tokenizer.eos_token_id), 1.0)
                    sel_pos = weights_on_bos.argmin(dim=1)
                
            else:
                arange_idx, sel_pos = choose_random_pos(args, tokens_batch, tokenizer)

            bos_ratio = 0.01
            sel_pos.masked_fill_(torch.rand_like(sel_pos, dtype=torch.float) < bos_ratio, 0)

            selected_act = activation[arange_idx, sel_pos]
            assert selected_act.dim() == 2
            try:
                write_q.put((selected_act.cpu(), tokens_batch.tolist(), probed_act), timeout=10)
            except queue.Full:
                print("writing queue is full, abort item")

    del hooked_model
    
    
def collect_test_data(args, device, probed_model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast, rollouts: rolloutManagerIOI, probed_acts: list[str]):
    batch_size = 10
    total_num = args.num_test_rollout
    total_step = -(total_num // -batch_size)  # ceiling version of //

    """ init hooked model on this device """
    hooked_model = init_hooked_model(args, device, probed_model, tokenizer)

    """ collect activation and corresponding rollout"""
    
    test_act_dataset = {probed_act: ActivationDataset(total_num, probed_acts) for probed_act in probed_acts}
    torch.manual_seed(99)
    with torch.no_grad():
        print("collecting activations...")
        if args.probed_task != "fact":
            cache = add_necessary_hooks(hooked_model, probed_acts)
            for _ in tqdm(range(total_step)):
                text_batch = rollouts.next_batch(batch_size)
                while torch.rand(()) < 0.8:
                    text_batch = rollouts.next_batch(batch_size)

                text_batch = list(map(lambda x: x + tokenizer.eos_token, text_batch))

                tokens_batch = hooked_model.to_tokens(text_batch)

                hooked_model(tokens_batch, return_type=None)

                for probed_act in probed_acts:
                    activation = retrieve_act(probed_act, cache)

                    arange_idx, sel_pos = choose_random_pos(args, tokens_batch, tokenizer)
                    selected_act = activation[arange_idx, sel_pos]
                    assert selected_act.dim() == 2
                    test_act_dataset[probed_act].add_data(selected_act.cpu(), tokens_batch.tolist(), probed_act)
            end_caching(hooked_model)

        else:
            for probed_act in probed_acts:
                cache = add_necessary_hooks(hooked_model, [probed_act])
                block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
                for _ in tqdm(range(total_step)):
                    text_batch = rollouts.next_batch(batch_size, probed_act, True)
                    while torch.rand(()) < 0.8:
                        text_batch = rollouts.next_batch(batch_size, probed_act, True)

                    text_batch_short = list(map(lambda x: x[0] + tokenizer.eos_token, text_batch))  # without last label 
                    sel_pos = (hooked_model.to_tokens(text_batch_short) == tokenizer.eos_token_id).float().argmax(dim=1) - 1 
                    text_batch = list(map(lambda x: x[1] + tokenizer.eos_token, text_batch)) 

                    tokens_batch = hooked_model.to_tokens(text_batch)

                    hooked_model(tokens_batch, stop_at_layer=block_idx+1)
                    activation = retrieve_act(probed_act, cache)

                    head_idx = int(re.search(r"hook_result\.(\d+)", probed_act).group(1)) 
                    weights_on_bos = cache[f"blocks.{block_idx}.attn.hook_pattern"][:, head_idx, :, 0].clone()
                    weights_on_bos.masked_fill_((tokens_batch == tokenizer.pad_token_id) | (tokens_batch == tokenizer.eos_token_id), 1.0)
                    sel_pos = weights_on_bos.argmin(dim=1)
                    

                    arange_idx = torch.arange(tokens_batch.size(0), dtype=torch.long, device=device)
                    selected_act = activation[arange_idx, sel_pos]

                    test_act_dataset[probed_act].add_data(selected_act.cpu(), tokens_batch.tolist(), probed_act)

                end_caching(hooked_model)               

    print("test dataset is created ")
    del rollouts
    del hooked_model

    return test_act_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="temp") # choices=["temp", "random", "named"]
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--data_per_epoch", type=int, default=1_000_000)
    parser.add_argument("--num_test_rollout", type=int, default=1_000)
    parser.add_argument("--pos_bias", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--caching_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--acc_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--arch_h", type=int, default=4)
    parser.add_argument("--arch_d", type=int, default=256)
    parser.add_argument("--arch_l", type=int, default=2)
    parser.add_argument("--gen_num", type=int, default=1)
    parser.add_argument("--cross_attn", action="store_true")
    parser.add_argument("--rebalance", type=float, default=0.0) # 6.0
    parser.add_argument("--probed_task", type=str, choices=["ioi", "addition", "counting", "fact"])
    parser.add_argument("--cossim", action="store_true")
    parser.add_argument("--no_saving", action="store_true")
    parser.add_argument("--pretrained", action="store_true")


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("device num", torch.cuda.device_count())
    num_threads = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(args)

    probed_model_path, data_path, exp_save_dir, max_len = get_paths(args)
    print("experiment directory:", exp_save_dir)
    args.max_len = max_len


    if args.probed_task == "addition":
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from addition.train import customTokenizer, make_dataset
        tokenizer = customTokenizer()
    elif args.probed_task == "counting":
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from counting.train import customTokenizer, make_dataset
        tokenizer = customTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(probed_model_path, add_bos_token=True)

    probed_model = GPT2LMHeadModel.from_pretrained(probed_model_path)
    print(probed_model.device)  # cpu

    
    if args.probed_task in ["ioi", "fact"]:
        tokenizer.add_special_tokens({"eos_token": "[EOS]", "pad_token": "[PAD]"})
        #  bos: <|endoftext|> 50256     eos: [EOS] 50257     pad: [PAD] 50258
        probed_model.resize_token_embeddings(probed_model.config.vocab_size+2)
        probed_model.config.bos_token_id = tokenizer.bos_token_id
        probed_model.config.eos_token_id = tokenizer.eos_token_id
        probed_model.config.pad_token_id = tokenizer.pad_token_id
    

    if args.probed_task == "ioi":
        probed_acts = ["blocks.0.hook_resid_pre"]
        for i in range(probed_model.config.n_layer):
            for j in range(probed_model.config.n_head):
                probed_acts.append(f"blocks.{i}.attn.hook_result.{j}")
            probed_acts.append(f"blocks.{i}.hook_mlp_out")
    elif args.probed_task == "counting":
        probed_acts = ["blocks.0.hook_resid_pre"]
        for i in range(probed_model.config.n_layer):
            for j in range(probed_model.config.n_head):
                probed_acts.append(f"blocks.{i}.attn.hook_result.{j}")
            probed_acts.append(f"blocks.{i}.hook_resid_mid")
            probed_acts.append(f"blocks.{i}.hook_mlp_out")
            probed_acts.append(f"blocks.{i}.hook_resid_post")
    elif args.probed_task == "addition":
        probed_acts = ["blocks.0.hook_resid_pre"]
        for i in range(probed_model.config.n_layer):
            for j in range(probed_model.config.n_head):
                probed_acts.append(f"blocks.{i}.attn.hook_result.{j}")
            probed_acts.append(f"blocks.{i}.hook_resid_mid")
            probed_acts.append(f"blocks.{i}.hook_resid_post")
    elif args.probed_task == "fact":
        probed_acts = []
        sel_heads = '31.0 33.0 38.22 29.9 37.7 32.12 31.8 24.24 28.3 25.7 28.21 27.16 42.24 31.4 34.20 30.23 24.8 30.8 25.10 33.9 32.15 30.1 29.20 36.17 35.19'
        for head in sel_heads.split(" "):
            layer_idx, head_idx = head.split(".")
            probed_acts.append(f"blocks.{layer_idx}.attn.hook_result.{head_idx}")
    

    if args.probed_task == "ioi":
        rollouts = rolloutManagerIOI(data_path)
        test_rollouts = rolloutManagerIOI(data_path)
    elif args.probed_task == "addition":
        rollouts = rolloutManagerAddition(make_dataset(tokenizer, train_ratio=1.0)[0])
        test_rollouts = rolloutManagerAddition(make_dataset(tokenizer, train_ratio=1.0)[0])
    elif args.probed_task == "counting":
        rollouts = rolloutManagerCounting(make_dataset(tokenizer, train_ratio=1.0)[0])
        test_rollouts = rolloutManagerCounting(make_dataset(tokenizer, train_ratio=1.0)[0])
    elif args.probed_task == "fact":
        rollouts = rolloutManagerFact(load_from_disk(data_path))
        test_rollouts = rolloutManagerFact(load_from_disk(data_path))

    data_collator = customCollator(tokenizer)

    act_dataset = ActivationDataset(args.data_per_epoch, probed_acts)

    read_lock = threading.Lock()
    write_q = queue.Queue(maxsize=50)
    handles = []
    for i in range(num_threads):
        t_device = torch.device(f"cuda:{i}") if torch.cuda.is_available() else torch.device("cpu")
        handle = threading.Thread(target=collect_data_thread, args=(args, t_device, read_lock, write_q, probed_model, tokenizer, act_dataset, rollouts, probed_acts, args.seed+i), daemon=True)
        handle.start()
        handles.append(handle)

    pbar = tqdm(total=act_dataset.data_per_epoch)
    print("collecting activations...")
    while len(act_dataset) < act_dataset.data_per_epoch:
        added_num = act_dataset.add_data(*write_q.get())
        pbar.update(added_num)
    pbar.close()

    for h in handles:
        h.join()

    print("dataset is created. ", len(act_dataset))

    test_act_dataset = collect_test_data(args, device, probed_model, tokenizer, test_rollouts, probed_acts)
    test_loader = {
        probed_act: DataLoader(test_act_dataset[probed_act], batch_size=args.caching_batch_size, shuffle=False, collate_fn=data_collator)
            for probed_act in probed_acts
    }

    act_proc_cfg = {
        "act_dim": act_dataset.activations.size(1),
        "probed_acts": probed_acts,
        "cross_attn": args.cross_attn,
        "act_proc_resid_dim": act_dataset.activations.size(1) + len(probed_acts),
        "act_proc_mid_dim": act_dataset.activations.size(1), 
        "act_proc_num_mlp": 6,
    }

    if args.pretrained:
        # load pretrained gpt2
        model = CustomGPT2LMHeadModel.from_pretrained("gpt2-medium", attn_pdrop=0, act_proc_cfg=act_proc_cfg)
        model.resize_token_embeddings(model.config.vocab_size+2)
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
    else:
        cfg = GPT2Config(vocab_size=len(tokenizer), 
                        n_positions=max_len+1,  # for eos
                        n_embd=args.arch_d,
                        n_layer=args.arch_l,
                        n_head=args.arch_h,
                        bos_token_id=tokenizer.bos_token_id, 
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        attn_pdrop=0.0,
                        )
        model = CustomGPT2LMHeadModel(cfg, act_proc_cfg=act_proc_cfg)
        print(model.config.bos_token_id)
        print(model.config.eos_token_id)
        print(model.config.pad_token_id)

    with torch.no_grad():
        for probed_act in probed_acts:
            all_act = []
            for input_batch in test_loader[probed_act]:
                all_act.append(input_batch["activation"].to(model.device))
            all_act = torch.cat(all_act, dim=0)
            old_emb = model.act_proc_layer_emb.weight.data[model.probed_acts.index(probed_act)]
            new_emb = (old_emb - old_emb.mean()) / old_emb.std() * all_act.std() + all_act.mean()
            model.act_proc_layer_emb.weight.data[model.probed_acts.index(probed_act)] = new_emb

    training_args = TrainingArguments(
        output_dir=exp_save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.acc_steps,
        evaluation_strategy='no',
        num_train_epochs=args.num_epoch,
        save_strategy="steps" if not args.no_saving else "no",
        save_steps=20/args.num_epoch,
        save_total_limit=4,
        logging_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        optim='adamw_torch',
        fp16=args.fp16,
        lr_scheduler_type='constant',
        report_to="none",
        torch_compile=args.compile,
        use_cpu=(not torch.cuda.is_available()),
    )


    class RegenerateAndGenerate(TrainerCallback):
        def on_epoch_begin(self, train_args: TrainingArguments, state: TrainerState, control: TrainerControl, model: CustomGPT2LMHeadModel, optimizer, **kwargs):
            if state.epoch > 0:
                # regenerate data in act_dataset
                act_dataset.re_init()

                handles = []
                for i in range(num_threads):
                    t_device = torch.device(f"cuda:{i}") if torch.cuda.is_available() else torch.device("cpu")
                    handle = threading.Thread(target=collect_data_thread, args=(args, t_device, read_lock, write_q, probed_model, tokenizer, act_dataset, rollouts, probed_acts, args.seed+i+int(state.epoch)*num_threads), daemon=True)
                    handle.start()
                    handles.append(handle)

                pbar = tqdm(total=act_dataset.data_per_epoch)
                print("collecting activations...")
                while len(act_dataset) < act_dataset.data_per_epoch:
                    added_num = act_dataset.add_data(*write_q.get())
                    pbar.update(added_num)
                pbar.close()

                for h in handles:
                    h.join()

                print("dataset is created. ", len(act_dataset))

        def on_epoch_end(self, train_args: TrainingArguments, state: TrainerState, control: TrainerControl, model: CustomGPT2LMHeadModel, **kwargs):
            if round(state.epoch) % 20 == 0:   # int(0.1 * args.num_epoch) 
                hooked_model = init_hooked_model(args, device, probed_model, tokenizer)

                model.eval()
                probed_act = probed_acts[torch.randint(0, len(probed_acts), ()).item()]
                print("\n", probed_act, "is selected")
                
                test_act = get_test_activation(args, probed_act, hooked_model)
                generate_and_print(args, test_act, model, tokenizer)

                prob_weight = run_test(args, probed_acts, test_loader, model, tokenizer, hooked_model, max_len)
                model.prob_weight = prob_weight
                model.train()

                del hooked_model


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=act_dataset,
        data_collator=data_collator,
        callbacks=[RegenerateAndGenerate],
    )

    trainer.train()