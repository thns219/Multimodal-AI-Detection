import os
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification

# =========================
# FIX PATH
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "data", "train.csv")
model_path = os.path.join(BASE_DIR, "models", "bert_detector")

# =========================
# LOAD DATA
# =========================

data = pd.read_csv(data_path)

human = data[["human_content"]].copy()
human.columns = ["text"]
human["label"] = 0

ai = data[["aigenerated_content_cleaned"]].copy()
ai.columns = ["text"]
ai["label"] = 1

dataset = pd.concat([human, ai])
dataset = dataset.dropna()

print("Dataset size:", dataset.shape)

# =========================
# SPLIT DATA
# =========================

X = dataset["text"]
y = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# LOAD MODEL
# =========================

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

# =========================
# PREDICT
# =========================

predictions = []

for text in X_test:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    predictions.append(pred)

# =========================
# METRICS
# =========================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n===== BERT PERFORMANCE =====\n")

print("Accuracy:", round(accuracy,3))
print("Precision:", round(precision,3))
print("Recall:", round(recall,3))
print("F1:", round(f1,3))