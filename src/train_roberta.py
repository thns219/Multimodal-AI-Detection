import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments


# ===============================
# FIX RANDOM
# ===============================
np.random.seed(42)
torch.manual_seed(42)


# ===============================
# LOAD DATA
# ===============================
print("Loading dataset...")
file = "../data/final_dataset.csv"
data = pd.read_csv(file)


# ===============================
# BUILD DATASET (🔥 FIX PARAPHRASE LEAK)
# ===============================
def build_full_dataset(data):
    human_texts = data["human_content"].dropna().tolist()
    ai_texts = data["aigenerated_content_cleaned"].dropna().tolist()

    # 🔥 shuffle riêng → phá cặp
    np.random.shuffle(human_texts)
    np.random.shuffle(ai_texts)

    rows = []

    for text in human_texts:
        rows.append({"text": text.strip().lower(), "label": 0})

    for text in ai_texts:
        rows.append({"text": text.strip().lower(), "label": 1})

    df = pd.DataFrame(rows)

    # remove duplicate
    df = df.drop_duplicates(subset=["text"])

    # remove text quá ngắn
    df = df[df["text"].str.len() > 50]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


full_df = build_full_dataset(data)

print("Total dataset:", len(full_df))
print(full_df["label"].value_counts())


# ===============================
# SPLIT CHUẨN
# ===============================
train_df, test_df = train_test_split(
    full_df,
    test_size=0.3,
    random_state=42,
    stratify=full_df["label"]
)


# LIMIT SIZE
train_df = train_df.sample(min(10000, len(train_df)), random_state=42)
test_df = test_df.sample(min(3000, len(test_df)), random_state=42)

print("Train:", len(train_df))
print("Test :", len(test_df))


# ===============================
# CHECK OVERLAP (PHẢI = 0)
# ===============================
overlap = set(train_df["text"]) & set(test_df["text"])
print("Overlap:", len(overlap))


# ===============================
# DATASET
# ===============================
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# ===============================
# TOKENIZER
# ===============================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
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
# TRAINING
# ===============================
training_args = TrainingArguments(
    output_dir="../models/roberta_detector",

    learning_rate=3e-5,
    num_train_epochs=2,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    weight_decay=0.1,
    logging_steps=100,

    do_train=True,
    do_eval=True,

    fp16=torch.cuda.is_available()
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# ===============================
# TRAIN
# ===============================
print("\nTRAINING...\n")
trainer.train()


# ===============================
# EVALUATE
# ===============================
print("\nEVALUATE...\n")
results = trainer.evaluate()

for k, v in results.items():
    print(f"{k}: {v:.4f}")


# ===============================
# CONFUSION MATRIX
# ===============================
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids

cm = confusion_matrix(labels, preds)

os.makedirs("../result", exist_ok=True)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - RoBERTa")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("../result/roberta_confusion_matrix.png")
plt.close()

print("Saved confusion matrix!")


# ===============================
# SAVE MODEL
# ===============================
trainer.save_model("../models/roberta_detector")
tokenizer.save_pretrained("../models/roberta_detector")

print("\nSaved model!")