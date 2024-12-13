#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
import argparse
import datasets
import transformers
import numpy as np
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from accelerate.utils import DistributedDataParallelKwargs
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description="Run model with specified checkpoint and CWE code")
parser.add_argument("model_checkpoint", type=str, help="Model checkpoint to use")
parser.add_argument("ntc", type=str, help="N Top or Cwe (NTC)")
parser.add_argument("datadir", type=str, help="Dataset's location")
parser.add_argument("bm", type=str, help="'b' for binary, 'm' for multiclass")
parser.add_argument("gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("train_batch", type=int, default=1, help="Number of batch(es) for training")
parser.add_argument("eval_batch", type=int, default=1, help="Number of batch(es) for evaluation")
parser.add_argument("tc", type=str, help="top or cwe")
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
ntc = args.ntc
datadir = args.datadir
bm = args.bm
gradient_accumulation_steps = args.gradient_accumulation_steps
train_batch = args.train_batch
eval_batch = args.eval_batch
tc = args.tc

# Add at the start of your script, after imports
def get_device_map():
    if dist.is_initialized():
        # For distributed training, use the local rank
        device = dist.get_rank() % torch.cuda.device_count()
    else:
        device = torch.cuda.current_device()
    print(f"Selected device: {device}")
    return {'': device}

# Create accelerator first, before model loading
def create_accelerator():
    if model_checkpoint == "bigcode/starcoder2-7b":
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        return Accelerator(
            device_placement=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="bf16",
            kwargs_handlers=[ddp_kwargs]
        )
    else:
        return Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="bf16"
        )

# Load datasets
if (tc == "top"):
    train_df = pd.read_json(f"{datadir}/pv_train_top_{ntc}.json")
    val_df = pd.read_json(f"{datadir}/pv_test_top_{ntc}.json")
    model_dir = "../../selective_model_RQ2_PV"
elif (tc == "cwe"):
    train_df = pd.read_json(f"{datadir}/pv_train_{ntc}.json")
    val_df = pd.read_json(f"{datadir}/pv_test_{ntc}.json")
    model_dir = "../../selective_model_RQ1_PV"
elif (tc == "all"):
    train_df = pd.read_json(f"{datadir}/{ntc}_date_train.json")
    val_df = pd.read_json(f"{datadir}/{ntc}_date_test.json")
    model_dir = "model"

print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

def apply_lora_and_quantization(model_checkpoint, tokenizer, num_labels, device_map):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        task_type="SEQ_CLS"
    )

    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["score"])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True
    )
    print(f"Initial model device: {next(model.parameters()).device}")
    
    model = get_peft_model(model, lora_config)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    print(f"Applied LoRA and quantization to the model.")
    print(f"Final model device: {next(model.parameters()).device}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    return model

if (ntc == "java"):
    train_df = train_df.drop(['index', 'committer_date'], axis=1)
    val_df = val_df.drop(['index', 'committer_date'], axis=1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(val_df)

raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})

config = AutoConfig.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model_max_length = getattr(config, 'max_position_embeddings', 512)  # Default to 512 if not set
if 'roberta' in config.model_type or 'bert' in config.model_type:
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    model_max_length -= num_special_tokens
max_seq_length = min(4020, model_max_length)
print(f"Using max_seq_length: {max_seq_length}")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    need_resize = True
else:
    need_resize = False

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,  # Ensure truncation
        padding="max_length",
        max_length=max_seq_length,
    )
    for idx, input_ids in enumerate(tokenized['input_ids']):
        if len(input_ids) > max_seq_length:
            print(f"Sequence at index {idx} exceeds max_seq_length")
    return tokenized

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

if 'label' in tokenized_datasets['train'].features:
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


def create_dataloaders(train_batch_size=1, eval_batch_size=1):
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, batch_size=eval_batch_size
    )
    return train_dataloader, eval_dataloader

train_dataloader, eval_dataloader = create_dataloaders()

hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "train_batch_size": train_batch,
    "eval_batch_size": train_batch,
    "seed": 42
}

accelerator = create_accelerator()
device_map = get_device_map()

if model_checkpoint == "bigcode/starcoder2-7b":
    model = apply_lora_and_quantization(model_checkpoint, tokenizer, num_labels=2, device_map=device_map)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=2,trust_remote_code=True).cuda()

# print(model)

if need_resize:
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()

def training_function(model):
    model.config.use_cache = False

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"],
        eval_batch_size=hyperparameters["eval_batch_size"]
    )
    set_seed(hyperparameters["seed"])
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    num_epochs = hyperparameters["num_epochs"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=len(train_dataloader) * num_epochs,
    )
    
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            try:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            except Exception as e:
                print(f"Error at step {step}: {e}")
                raise e

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"{model_dir}/{model_checkpoint}_{ntc}_{epoch}")
            tokenizer.save_pretrained(f"{model_dir}/{model_checkpoint}_{ntc}_{epoch}")

training_function(model)
