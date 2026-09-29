import sys
import re
from datetime import datetime, timezone

import joblib

vec, clf = joblib.load("tfidf_model.joblib")
vocab = vec.vocabulary_
CLASSES = list(clf.classes_)

try:
    from pymongo import MongoClient
    log_col = MongoClient("mongodb://localhost:27017",
                          serverSelectionTimeoutMS=1000)["news_db"]["predictions"]
except Exception:
    log_col = None


def show(text):
    words = re.findall(r"\b\w\w+\b", text.lower())
    known = [w for w in words if w in vocab]
    if not known:
        print("\nNo recognizable news vocabulary found. Try a longer, news-style text.")
        return

    probs = clf.predict_proba(vec.transform([text]))[0]
    order = probs.argsort()[::-1]
    best, second = order[0], order[1]
    margin = probs[best] - probs[second]

    print(f"\nCategory  : {CLASSES[best].upper()}")
    print(f"Confidence: {probs[best]:.1%}")
    if margin < 0.10:
        print(f"Note: close call between {CLASSES[best]} and {CLASSES[second]}.")
    if len(words) < 10:
        print("Note: short text, longer articles give more reliable results.")
    if len(known) / len(words) < 0.5:
        print("Note: many words are unfamiliar to the model, so treat this with caution.")
    print("All scores:")
    for i in order:
        print(f"  {CLASSES[i]:<14}{probs[i]:.1%}")

    if log_col is not None:
        try:
            log_col.insert_one({
                "text": text[:2000],
                "predicted": CLASSES[best],
                "confidence": float(probs[best]),
                "model": "tfidf_logreg",
                "time": datetime.now(timezone.utc),
            })
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        with open(sys.argv[2], encoding="utf-8") as f:
            show(f.read())
    elif len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        print("Type or paste one article on a single line. Press Enter on an empty line to quit.")
        while True:
            text = input("\n> ").strip()
            if not text:
                break
            show(text)