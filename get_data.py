from datasets import load_dataset
import pandas as pd

ds = load_dataset("SetFit/bbc-news")
df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)

df = df.rename(columns={"label_text": "category"})
df = df[["category", "text"]]
df.to_csv("bbc-text.csv", index=False)

print(df.shape)
print(df["category"].value_counts())