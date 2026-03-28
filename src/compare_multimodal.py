import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
RESULT_DIR = os.path.join(ROOT_DIR, "result")

# ===============================
# LOAD METRICS CHUẨN
# ===============================
def load_metrics(path):
    df = pd.read_csv(path, index_col=0)

    accuracy = df.loc["accuracy", "f1-score"] if "accuracy" in df.index else df.iloc[-1, 0]
    precision = df.loc["macro avg", "precision"]
    recall = df.loc["macro avg", "recall"]
    f1 = df.loc["macro avg", "f1-score"]

    return [accuracy, precision, recall, f1]

bert = load_metrics(os.path.join(RESULT_DIR, "bert_report.csv"))
roberta = load_metrics(os.path.join(RESULT_DIR, "roberta_report.csv"))
vit = load_metrics(os.path.join(RESULT_DIR, "vit_report.csv"))

# ===============================
# CREATE TABLE
# ===============================
df_compare = pd.DataFrame({
    "Accuracy": [bert[0], roberta[0], vit[0]],
    "Precision": [bert[1], roberta[1], vit[1]],
    "Recall": [bert[2], roberta[2], vit[2]],
    "F1": [bert[3], roberta[3], vit[3]]
}, index=["BERT", "RoBERTa", "ViT"])

# ===============================
# SAVE CSV + TXT
# ===============================
df_compare.to_csv(os.path.join(RESULT_DIR, "compare_multimodal.csv"))

with open(os.path.join(RESULT_DIR, "compare_multimodal.txt"), "w") as f:
    f.write(df_compare.to_string())

# ===============================
# IEEE CLEAN CHART (FINAL)
# ===============================
plt.close('all')  # fix lỗi chồng ảnh

plt.style.use("default")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
ax.set_facecolor("white")

x = np.arange(len(df_compare.columns))
width = 0.22

colors = ["#4C72B0", "#55A868", "#C44E52"]

ax.bar(x - width, df_compare.loc["BERT"], width, label="BERT", color=colors[0])
ax.bar(x, df_compare.loc["RoBERTa"], width, label="RoBERTa", color=colors[1])
ax.bar(x + width, df_compare.loc["ViT"], width, label="ViT", color=colors[2])

# Axis
ax.set_xticks(x)
ax.set_xticklabels(df_compare.columns, fontsize=11)
ax.set_ylim(0, 1.05)

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison", fontsize=13, weight="bold")

# Clean style
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 🔥 LEGEND TRÊN (ĐẸP NHẤT)
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3
)

# layout chuẩn paper
plt.tight_layout(rect=[0, 0, 1, 0.9])

# SAVE
plt.savefig(
    os.path.join(RESULT_DIR, "compare.png"),
    dpi=300,
    facecolor="white",
    bbox_inches="tight"
)

plt.close('all')

print("\n✅ DONE")