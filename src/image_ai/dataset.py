import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    data = []

    for label, folder in enumerate(["0_real", "1_fake"]):
        folder_path = os.path.join(data_dir, folder)

        for file in os.listdir(folder_path):
            if file.endswith((".jpg", ".png", ".jpeg")):
                data.append([os.path.join(folder_path, file), label])

    df = pd.DataFrame(data, columns=["image", "label"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    return train_df, test_df