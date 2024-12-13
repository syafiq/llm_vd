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
import json
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
    T5Config, 
    T5ForConditionalGeneration,
    RobertaTokenizer,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from model import DefectModel, Model

# Environment settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define model classes
MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'roberta': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}

def compute_metrics(labels, preds):
    """Compute metrics for binary classification"""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'fpr': fpr * 100,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

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

# Set default values for required attributes
args.local_rank = int(os.getenv('LOCAL_RANK', -1))
args.no_cuda = not torch.cuda.is_available()
args.output_dir = "output"
args.train_batch_size = args.train_batch
args.eval_batch_size = args.eval_batch
args.seed = 42
args.project = "default"
args.model_dir = "model"

model_checkpoint = args.model_checkpoint
model_name = model_checkpoint.split("/")[-1]
ntc = args.ntc
datadir = args.datadir
bm = args.bm
gradient_accumulation_steps = args.gradient_accumulation_steps
train_batch = args.train_batch
eval_batch = args.eval_batch
tc = args.tc

def get_device_map():
    if dist.is_initialized():
        device = dist.get_rank() % torch.cuda.device_count()
    else:
        device = torch.cuda.current_device()
    print(f"Selected device: {device}")
    return {'': device}

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
    all_df = pd.read_json(f"{datadir}/rest/pv_TOP_test_all_{ntc}.json")
    model_dir = "../../selective_model_RQ2_PV"
elif (tc == "cwe"):
    train_df = pd.read_json(f"{datadir}/pv_train_{ntc}.json")
    val_df = pd.read_json(f"{datadir}/pv_test_{ntc}.json")
    all_df = pd.read_json(f"{datadir}/rest/pv_test_all_{ntc}.json")
    model_dir = "../../selective_model_RQ1_PV"
elif (tc == "all"):
    train_df = pd.read_json(f"{datadir}/{ntc}_date_train.json")
    val_df = pd.read_json(f"{datadir}/{ntc}_date_test.json")
    model_dir = "model"

print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

def initialize_model(model_checkpoint, config, tokenizer, device_map):
    if "codet5" in model_checkpoint.lower():
        config_class, model_class, _ = MODEL_CLASSES['codet5']
        base_model = model_class.from_pretrained(
            model_checkpoint,
            config=config,
            trust_remote_code=True
        )
        model = DefectModel(base_model, config, tokenizer, args)
    else:
        model = Model(
            AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=2,
                trust_remote_code=True
            ),
            config,
            tokenizer,
            args
        ).cuda()
    
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    
    return model

#train_df = train_df.drop(['year', 'cwe', 'source', 'hash'], axis=1)
#val_df = val_df.drop(['year', 'cwe', 'source', 'hash'], axis=1)
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

model_max_length = getattr(config, 'max_position_embeddings', 512)
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
    """Tokenize and validate <eos> token consistency."""
    valid_texts = []
    valid_labels = []  # Ensure alignment of labels with texts

    for text, label in zip(examples["text"], examples["label"]):
        input_ids = tokenizer.encode(
            text,
            truncation=True,
            max_length=max_seq_length,
        )
        eos_count = input_ids.count(tokenizer.eos_token_id)

        if eos_count == 1:  # Only keep valid rows with exactly one <eos>
            valid_texts.append(text)
            valid_labels.append(label)

    tokenized = tokenizer(
        valid_texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    tokenized["labels"] = valid_labels  # Add the aligned labels back
    return tokenized

# Tokenize and validate dataset in one step
logger.info("Starting tokenization and validation...")
logger.info(f"Original training dataset size: {len(raw_datasets['train'])}")
logger.info(f"Original validation dataset size: {len(raw_datasets['validation'])}")

# Tokenize and validate dataset in one step
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text", "label"])

# Set dataset format
if "label" in tokenized_datasets["train"].features:
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

logger.info(f"Tokenized training dataset size: {len(tokenized_datasets['train'])}")
logger.info(f"Tokenized validation dataset size: {len(tokenized_datasets['validation'])}")

def create_dataloaders(train_batch_size=1, eval_batch_size=1):
    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        shuffle=False, 
        batch_size=eval_batch_size
    )
    return train_dataloader, eval_dataloader

def evaluate_and_save(model, dataloader, accelerator, save_dir, prefix):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["input_ids"], batch["labels"])
            if isinstance(outputs, tuple):
                _, logits = outputs
            else:
                logits = outputs
            
            predictions = (logits[:, 1] > 0.5 if logits.shape[1] == 2 else logits > 0.5).long()
            all_predictions.extend(accelerator.gather(predictions).cpu().numpy())
            all_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy())
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save predictions and labels separately
    pred_path = os.path.join(save_dir, f"{prefix}_predictions.pt")
    label_path = os.path.join(save_dir, f"{prefix}_labels.pt")
    torch.save(all_predictions, pred_path)
    torch.save(all_labels, label_path)
    logger.info(f"Saved predictions to {pred_path} and labels to {label_path}")
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions)
    return metrics

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
    
    best_f1 = 0
    best_model_path = None  # To track the best model path
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_predictions = []
        train_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            try:
                loss, logits = model(batch["input_ids"], batch["labels"])
                accelerator.backward(loss)
                
                total_loss += loss.item()
                predictions = (logits[:, 1] > 0.5 if logits.shape[1] == 2 else logits > 0.5).long()
                train_predictions.extend(accelerator.gather(predictions).cpu().numpy())
                train_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy())
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({'loss': total_loss / (step + 1)})
                
            except Exception as e:
                print(f"Error at step {step}: {e}")
                raise e

        # Compute training metrics
        train_metrics = compute_metrics(train_labels, train_predictions)
        
        # Evaluate on val_df and save predictions/labels
        eval_metrics = evaluate_and_save(
            model, eval_dataloader, accelerator,
            save_dir=f"pred_label",
            prefix=f"{model_name}_val_{ntc}_epoch_{epoch+1}"
        )
        
        # Log metrics
        if accelerator.is_main_process:
            logger.info(f"\n{'='*50}\nEpoch {epoch+1}:")
            logger.info("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                logger.info(f"{metric}: {value:.2f}")
            
            logger.info("\nValidation Metrics:")
            for metric, value in eval_metrics.items():
                logger.info(f"{metric}: {value:.2f}")
            
            # Save best model
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                logger.info(f"\nNew best F1 score: {best_f1:.2f}")
                
                # Save best model
                if (tc == "top"):
                    checkpoint_prefix = f'checkpoint-best-f1/{model_checkpoint}_top_{ntc}'
                if (tc == "cwe"):
                    checkpoint_prefix = f'checkpoint-best-f1/{model_checkpoint}_cwe_{ntc}'
                if (tc == "all"):
                    checkpoint_prefix = f'checkpoint-best-f1/{model_checkpoint}_cv_{ntc}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                os.makedirs(output_dir, exist_ok=True)
                
                # Save model using state dict
                unwrapped_model = accelerator.unwrap_model(model)
                model_path = os.path.join(output_dir, 'model.bin')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                }, model_path)
                best_model_path = model_path
                
                # Save tokenizer and config
                tokenizer.save_pretrained(output_dir)
                if hasattr(unwrapped_model, 'config'):
                    unwrapped_model.config.save_pretrained(output_dir)
                
                logger.info(f"Saved model checkpoint to {output_dir}")
    
    return best_model_path

hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "train_batch_size": train_batch,
    "eval_batch_size": eval_batch,
    "seed": 42
}

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)

    # Initialize accelerator and model
    accelerator = create_accelerator()
    device_map = get_device_map()

    # Set model type for CodeT5
    if "codet5" in model_checkpoint.lower():
        args.model_type = 'codet5'
        config.num_labels = 2
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize model
    model = initialize_model(model_checkpoint, config, tokenizer, device_map)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Log model info
    logger.info("Training/evaluation parameters:")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    logger.info("Starting training...")
    try:
        best_model_path = training_function(model)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise e

    # Load the best model for evaluation on all_df
    if ((tc == "top") or (tc == "cwe")):
        if best_model_path:
            logger.info("Loading the best model for evaluation on all_df...")
            model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
            model = accelerator.prepare(model)

            logger.info("Running evaluation on all_df...")
            all_dataset = Dataset.from_pandas(all_df)
            tokenized_all_dataset = all_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
            tokenized_all_dataset.set_format("torch")
            all_dataloader = DataLoader(
                tokenized_all_dataset,
                shuffle=False,
                batch_size=eval_batch
            )
            all_dataloader = accelerator.prepare(all_dataloader)

            all_metrics = evaluate_and_save(
                model,
                all_dataloader,
                accelerator,
                save_dir="pred_label",
                prefix=f"{model_name}_all_{ntc}_best"
            )

            logger.info("\nAll Dataset Metrics:")
            for metric, value in all_metrics.items():
                logger.info(f"{metric}: {value:.2f}")

