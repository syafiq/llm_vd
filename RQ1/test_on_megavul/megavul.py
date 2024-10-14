import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import sys

model="java"

c_test = pd.read_json(f"megavul_all.json")
c_test = c_test.reset_index(drop=True)
test_dataset = Dataset.from_pandas(c_test)

model_checkpoint = f"../model/{model}"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=4000)
    return outputs

tokenized_datasets = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")

test_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=1)

trained_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
trained_model.resize_token_embeddings(len(tokenizer))

trained_model.eval()
all_predictions = []
all_labels = []
trained_model = trained_model.cuda()

for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        # Move the input data to GPU
        batch = {k: v.cuda() for k, v in batch.items()}

        outputs = trained_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    all_predictions.append(predictions.cpu())  # Move predictions back to CPU
    all_labels.append(batch["labels"].cpu())  # Move labels back to CPU

all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)
torch.save(all_predictions, f"predictions_{model}_all.pt")
torch.save(all_labels, f"labels_{model}_all.pt")
