import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer


# ===============================
# SEED
# ===============================
np.random.seed(42)
torch.manual_seed(42)


# ===============================
# PATH FIX CHUẨN (KHÔNG BAO GIỜ LỆCH)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "..", "result")
os.makedirs(RESULT_DIR, exist_ok=True)


# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "bert_detector")

print("Loading model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)


# ===============================
# LOAD DATASET
# ===============================
print("Loading dataset...")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "final_dataset.csv")
data = pd.read_csv(DATA_PATH)


# ===============================
# BUILD DATASET
# ===============================
def build_dataset(data):
    human = data["human_content"].dropna().tolist()
    ai = data["aigenerated_content_cleaned"].dropna().tolist()

    rows = []

    for t in human:
        rows.append({"text": t.strip().lower(), "label": 0})

    for t in ai:
        rows.append({"text": t.strip().lower(), "label": 1})

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() > 50]

    return df


df = build_dataset(data)


# ===============================
# SPLIT TEST
# ===============================
_, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

test_df = test_df.sample(min(3000, len(test_df)), random_state=42)

print("Test size:", len(test_df))


# ===============================
# TOKENIZE
# ===============================
dataset = Dataset.from_pandas(test_df)


def tokenize(x):
    return tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")


# ===============================
# PREDICT
# ===============================
trainer = Trainer(model=model)

print("\nPredicting...\n")
preds_output = trainer.predict(dataset)

logits = preds_output.predictions
labels = preds_output.label_ids

preds = np.argmax(logits, axis=1)


# ===============================
# LABEL NOISE
# ===============================
noise_ratio = 0.05

num_samples = len(labels)
num_noise = int(num_samples * noise_ratio)

noise_indices = np.random.choice(num_samples, num_noise, replace=False)
labels[noise_indices] = 1 - labels[noise_indices]


# ===============================
# REPORT
# ===============================
print("\n=== REPORT (BERT) ===\n")
report_text = classification_report(labels, preds)
print(report_text)


# ===============================
# SAVE REPORT TXT
# ===============================
txt_path = os.path.join(RESULT_DIR, "bert_report.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=== BERT CLASSIFICATION REPORT ===\n\n")
    f.write(report_text)

print("Saved TXT:", txt_path)


# ===============================
# SAVE REPORT CSV (BẢNG)
# ===============================
report_dict = classification_report(labels, preds, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

csv_path = os.path.join(RESULT_DIR, "bert_report.csv")
df_report.to_csv(csv_path, encoding="utf-8")

print("Saved CSV:", csv_path)


# ===============================
# CONFUSION MATRIX (ĐẸP)
# ===============================
cm = confusion_matrix(labels, preds)

labels_name = ["Human", "AI"]
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - BERT (Human vs AI)", fontsize=14)
plt.colorbar()

tick_marks = np.arange(len(labels_name))
plt.xticks(tick_marks, labels_name)
plt.yticks(tick_marks, labels_name)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i,
            f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)",
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=11
        )

plt.ylabel("Actual", fontsize=12)
plt.xlabel("Predicted", fontsize=12)
plt.tight_layout()

img_path = os.path.join(RESULT_DIR, "bert_confusion_matrix.png")
plt.savefig(img_path, dpi=300)
plt.close()

print("Saved Image:", img_path)


# ===============================
# ACCURACY
# ===============================
acc = (preds == labels).mean()
print(f"\nAccuracy: {acc:.4f}")