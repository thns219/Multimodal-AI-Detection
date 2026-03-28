import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import ViTForImageClassification, ViTImageProcessor

from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# SEED
# ===============================
np.random.seed(42)
torch.manual_seed(42)

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "diffusion_data")
RESULT_DIR = os.path.join(BASE_DIR, "result")

os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# DATASET
# ===============================
class ImageDataset(Dataset):
    def __init__(self, paths, labels, processor):
        self.paths = paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        encoding = self.processor(images=img, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ===============================
# LOAD DATA
# ===============================
print("Loading dataset...")

image_paths = []
labels = []

label_map = {
    "0_real": 0,
    "1_fake": 1
}

for label_name, label_id in label_map.items():
    folder = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(folder, file))
            labels.append(label_id)

print("Total samples:", len(image_paths))

# ===============================
# DATA LOADER
# ===============================
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

dataset = ImageDataset(image_paths, labels, processor)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load(os.path.join(BASE_DIR, "vit_best.pth"), map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===============================
# PREDICT
# ===============================
print("\nPredicting...\n")

preds = []
labels_list = []

with torch.no_grad():
    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels_batch = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        pred = torch.argmax(outputs.logits, dim=1)

        preds.extend(pred.cpu().numpy())
        labels_list.extend(labels_batch.cpu().numpy())

# ===============================
# REPORT
# ===============================
print("\n=== REPORT (ViT) ===\n")

report_text = classification_report(labels_list, preds, target_names=["Real", "Fake"])
print(report_text)

# ===============================
# SAVE TXT
# ===============================
txt_path = os.path.join(RESULT_DIR, "vit_report.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=== VIT CLASSIFICATION REPORT ===\n\n")
    f.write(report_text)

print("Saved TXT:", txt_path)

# ===============================
# SAVE CSV
# ===============================
report_dict = classification_report(labels_list, preds, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

csv_path = os.path.join(RESULT_DIR, "vit_report.csv")
df_report.to_csv(csv_path, encoding="utf-8")

print("Saved CSV:", csv_path)

# ===============================
# CONFUSION MATRIX (ĐẸP NHƯ ROBERTA)
# ===============================
cm = confusion_matrix(labels_list, preds)

labels_name = ["Real", "Fake"]
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - ViT (Real vs Fake)", fontsize=14)
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

img_path = os.path.join(RESULT_DIR, "vit_confusion_matrix.png")
plt.savefig(img_path, dpi=300)
plt.close()

print("Saved Image:", img_path)

# ===============================
# ACCURACY
# ===============================
acc = (np.array(preds) == np.array(labels_list)).mean()
print(f"\nAccuracy: {acc:.4f}")