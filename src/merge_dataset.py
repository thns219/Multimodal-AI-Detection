import pandas as pd

files = [
    "../data/CoAID/train.csv",
    "../data/CoAID/test.csv",
    "../data/CoAID/validation.csv",

    "../data/FakeNewsNet_Gossipcop/test.csv",
    "../data/FakeNewsNet_Gossipcop/validation.csv",

    "../data/FakeNewsNet_Politifacts/train.csv",
    "../data/FakeNewsNet_Politifacts/test.csv",
    "../data/FakeNewsNet_Politifacts/validation.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("../data/final_dataset.csv", index=False)

print("Dataset merged successfully")
print("Total samples:", len(merged))