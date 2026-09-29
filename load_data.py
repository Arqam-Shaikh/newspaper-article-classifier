import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
col = client["news_db"]["articles"]

df = pd.read_csv("bbc-text.csv")
df = df.drop_duplicates(subset="text")

col.delete_many({})
col.insert_many(df.to_dict("records"))
col.create_index("category")

print("Total articles:", col.count_documents({}))
print(list(col.aggregate([{"$group": {"_id": "$category", "n": {"$sum": 1}}}])))