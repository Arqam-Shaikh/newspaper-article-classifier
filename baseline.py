import joblib
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

col = MongoClient("mongodb://localhost:27017")["news_db"]["articles"]
docs = list(col.find({}, {"text": 1, "category": 1}))
texts = [d["text"] for d in docs]
y = [d["category"] for d in docs]

# Same split as train.py (same random_state), so the comparison is fair
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    texts, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(X_tr)
Xval = vec.transform(X_val)

# Tune regularization strength C on the validation set (not the test set)
best_c, best_f1 = None, -1
for C in [1, 10, 30, 100]:
    m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_tr)
    f1 = f1_score(y_val, m.predict(Xval), average="macro")
    print(f"C={C:<4} validation macro F1 = {f1:.4f}")
    if f1 >= best_f1:          # on ties, prefer the larger C
        best_c, best_f1 = C, f1
print("Chosen C:", best_c)

clf = LogisticRegression(C=best_c, max_iter=2000).fit(Xtr, y_tr)

pred = clf.predict(vec.transform(X_te))
print("Test accuracy:", round(accuracy_score(y_te, pred), 4))
print("Macro F1:", round(f1_score(y_te, pred, average="macro"), 4))
print(classification_report(y_te, pred))

joblib.dump((vec, clf), "tfidf_model.joblib")
print("Saved tfidf_model.joblib")