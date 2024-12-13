#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import datasets
import transformers
import torch.distributed as dist
import random
import torch.distributed as dist
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from IPython.display import display, HTML
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    get_scheduler,
)

datadir  ="js" # change accordingly
shortname="js" # change accordingly

model_checkpoint = "deepseek-ai/deepseek-coder-1.3b-instruct"

train_df = pd.read_json(f"{datadir}/js_date_train.json") # change accordingly
val_df = pd.read_json(f"{datadir}/js_date_test.json") # change accordingly
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(val_df)

raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=4000)
    return outputs

tokenize_function(raw_datasets['train'][:5])
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets["train"].features
tokenized_datasets.set_format("torch")
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.resize_token_embeddings(len(tokenizer))

def create_dataloaders(train_batch_size=1, eval_batch_size=1):
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, batch_size=eval_batch_size
    )
    return train_dataloader, eval_dataloader


train_dataloader, eval_dataloader = create_dataloaders()

metric_accuracy = load_metric("accuracy")
metric_f1 = load_metric("f1")

# Add this to your hyperparameters
hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "train_batch_size": 1,  # Slightly lower batch size per GPU
    "eval_batch_size": 1,
    "seed": 42,
    "gradient_accumulation_steps": 8  # Accumulate gradients to simulate a larger batch size
}

def training_function(model):
    #accelerator = Accelerator(gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"], mixed_precision="fp16")
    accelerator = Accelerator(gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"])
    #accelerator = Accelerator()

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )
    set_seed(hyperparameters["seed"])
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    model.gradient_checkpointing_enable()

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    #print("Memory usage after initialization:")
    #torch.cuda.memory_summary(device=0)
    
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
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            #print(f"Memory usage after step {step}:")
            #torch.cuda.memory_summary(device=0)
            torch.cuda.empty_cache()

        model.eval()
        all_predictions = []
        all_labels = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            
            all_predictions.append(accelerator.gather(predictions))
            all_labels.append(accelerator.gather(batch["labels"]))

        all_predictions = torch.cat(all_predictions)[:len(tokenized_datasets["validation"])]
        all_labels = torch.cat(all_labels)[:len(tokenized_datasets["validation"])]
        print(f"Length of all_predictions: {len(all_predictions)}")
        print(f"Length of all_labels: {len(all_labels)}")
        torch.save(all_predictions, f"predictions_{shortname}_{epoch}.pt")
        torch.save(all_labels, f"labels_{shortname}_{epoch}.pt")

        eval_accuracy = metric_accuracy.compute(predictions=all_predictions, references=all_labels)["accuracy"]
        eval_f1 = metric_f1.compute(predictions=all_predictions, references=all_labels)["f1"]
        precision = precision_score(all_labels.cpu(), all_predictions.cpu())
        recall = recall_score(all_labels.cpu(), all_predictions.cpu())
        tn, fp, fn, tp = confusion_matrix(all_labels.cpu(), all_predictions.cpu()).ravel()
        fpr = fp / (fp + tn)

        print(f"  Epoch {epoch}:")
        print(f"  Accuracy: {eval_accuracy}")
        print(f"  F1 Score: {eval_f1}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  FPR: {fpr}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        if dist.get_rank() == 0:
            model.module.save_pretrained(f"{shortname}_model_{epoch}")
            tokenizer.save_pretrained(f"{shortname}_model_{epoch}")

training_function(model)
