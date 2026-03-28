import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments


# ===============================
# CHECK GPU
# ===============================
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))


print("Loading dataset...")

file = "../data/final_dataset.csv"
data_raw = pd.read_csv(file)


# ===============================
# SPLIT TRƯỚC (QUAN TRỌNG)
# ===============================
train_raw, test_raw = train_test_split(
    data_raw,
    test_size=0.3,
    random_state=42
)


# ===============================
# FORMAT (FIX DATA LEAK 100%)
# ===============================
def format_dataset(data):

    rows = []

    for _, row in data.iterrows():

        # mỗi row chỉ lấy 1 loại (human hoặc AI)
        if np.random.rand() < 0.5:
            if pd.notna(row["human_content"]):
                rows.append({
                    "text": row["human_content"],
                    "label": 0
                })
        else:
            if pd.notna(row["aigenerated_content_cleaned"]):
                rows.append({
                    "text": row["aigenerated_content_cleaned"],
                    "label": 1
                })

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["text"])
    df = df.sample(frac=1).reset_index(drop=True)

    return df


train_df = format_dataset(train_raw)
test_df = format_dataset(test_raw)


# ===============================
# 🔥 DATASET 10K
# ===============================
train_df = train_df.sample(min(10000, len(train_df)))
test_df = test_df.sample(min(3000, len(test_df)))

print("Train:", len(train_df))
print("Test :", len(test_df))


# ===============================
# DATASET
# ===============================
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# ===============================
# TOKENIZE
# ===============================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(x):
    return tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# ===============================
# MODEL
# ===============================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# ===============================
# METRICS
# ===============================
def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ===============================
# TRAINING (GPU AUTO)
# ===============================
training_args = TrainingArguments(
    output_dir="../models/bert_detector",

    learning_rate=5e-5,
    num_train_epochs=2,  # 🔥 tăng nhẹ cho học tốt hơn

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    weight_decay=0.2,

    logging_steps=100,

    do_train=True,
    do_eval=True,

    dataloader_pin_memory=True,   # 🔥 bật lại để tận dụng GPU

    fp16=torch.cuda.is_available()  # 🔥 tự bật nếu có GPU
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


print("\nTRAINING...\n")
trainer.train()


print("\nEVALUATE...\n")
results = trainer.evaluate()

for k, v in results.items():
    print(f"{k}: {v:.4f}")


trainer.save_model("../models/bert_detector")
tokenizer.save_pretrained("../models/bert_detector")

print("\nSaved model!")