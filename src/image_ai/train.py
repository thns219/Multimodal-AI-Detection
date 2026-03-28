import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import ViTForImageClassification, ViTImageProcessor

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ========================
# DATASET
# ========================
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

# ========================
# MAIN
# ========================
def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.path.join(BASE_DIR, "diffusion_data")
    RESULT_DIR = os.path.join(BASE_DIR, "result")

    os.makedirs(RESULT_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16

    # ========================
    # LOAD DATA (FULL TEST)
    # ========================
    image_paths = []
    labels = []

    label_map = {
        "0_real": 0,
        "1_fake": 1
    }

    for label_name, label_id in label_map.items():
        folder = os.path.join(DATA_DIR, label_name)

        if not os.path.exists(folder):
            raise ValueError(f"❌ Missing folder: {folder}")

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder, file))
                labels.append(label_id)

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    dataset = ImageDataset(image_paths, labels, processor)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ========================
    # LOAD MODEL
    # ========================
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "vit_best.pth"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ========================
    # PREDICT
    # ========================
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

    # ========================
    # METRICS
    # ========================
    acc = accuracy_score(labels_list, preds)
    precision = precision_score(labels_list, preds)
    recall = recall_score(labels_list, preds)
    f1 = f1_score(labels_list, preds)

    report = classification_report(labels_list, preds)
    print("\n=== TEST REPORT (ViT) ===\n")
    print(report)

    # ========================
    # SAVE TXT
    # ========================
    txt_path = os.path.join(RESULT_DIR, "vit_test_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}")
        f.write(f"\nPrecision: {precision:.4f}")
        f.write(f"\nRecall: {recall:.4f}")
        f.write(f"\nF1-score: {f1:.4f}")

    # ========================
    # SAVE CSV
    # ========================
    df = pd.DataFrame(classification_report(labels_list, preds, output_dict=True)).transpose()
    df.to_csv(os.path.join(RESULT_DIR, "vit_test_report.csv"))

    # ========================
    # CONFUSION MATRIX
    # ========================
    cm = confusion_matrix(labels_list, preds)
    labels_name = ["Real", "Fake"]
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black"
            )

    plt.xticks([0,1], labels_name)
    plt.yticks([0,1], labels_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - ViT")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "vit_test_confusion_matrix.png"), dpi=300)
    plt.close()

    # ========================
    # SUMMARY
    # ========================
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Value": [acc, precision, recall, f1]
    })

    summary.to_csv(os.path.join(RESULT_DIR, "vit_test_summary.csv"), index=False)

    print("\n✅ TEST DONE - Saved to result/")

# ========================
# RUN
# ========================
if __name__ == "__main__":
    main()