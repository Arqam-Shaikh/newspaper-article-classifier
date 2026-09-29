import re
import json
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pymongo import MongoClient, UpdateOne

torch.manual_seed(42)

# ---------- 1. Load articles from MongoDB ----------
client = MongoClient("mongodb://localhost:27017")
col = client["news_db"]["articles"]

docs = list(col.find({}, {"text": 1, "category": 1}))
labels = sorted({d["category"] for d in docs})
label2id = {l: i for i, l in enumerate(labels)}

texts = [d["text"] for d in docs]
y = [label2id[d["category"]] for d in docs]
ids = [d["_id"] for d in docs]
print("Articles:", len(docs), "| Classes:", labels)

# ---------- 2. Stratified split: 70% train / 15% val / 15% test ----------
X_tr, X_tmp, y_tr, y_tmp, id_tr, id_tmp = train_test_split(
    texts, y, ids, test_size=0.3, stratify=y, random_state=42)
X_val, X_te, y_val, y_te, id_val, id_te = train_test_split(
    X_tmp, y_tmp, id_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
print("Train/Val/Test:", len(X_tr), len(X_val), len(X_te))

# ---------- 3. Tokenize and build vocabulary (train set only) ----------
def tokenize(t):
    return re.findall(r"[a-z']+", t.lower())

counter = Counter(w for t in X_tr for w in tokenize(t))
vocab = {"<pad>": 0, "<unk>": 1}
for w, c in counter.items():
    if c >= 3:                      # drop very rare words
        vocab[w] = len(vocab)
print("Vocab size:", len(vocab))

MAX_LEN = 300
def encode(t):
    ix = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN]
    return ix + [0] * (MAX_LEN - len(ix))

# ---------- 4. PyTorch Dataset and DataLoaders ----------
class NewsDS(Dataset):
    def __init__(self, texts, labels):
        self.x = torch.tensor([encode(t) for t in texts])
        self.y = torch.tensor(labels)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_dl = DataLoader(NewsDS(X_tr, y_tr), batch_size=32, shuffle=True)
val_dl = DataLoader(NewsDS(X_val, y_val), batch_size=64)
test_dl = DataLoader(NewsDS(X_te, y_te), batch_size=64)

# ---------- 5. Model: Embedding -> BiLSTM -> masked mean pool -> Dropout -> Linear ----------
class Classifier(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=128, n_classes=5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * hid, n_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                      # ignore padding
        out, _ = self.lstm(self.emb(x))
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier(len(vocab), n_classes=len(labels)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

# ---------- 6. Train / validate ----------
def run(dl, train=False):
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot_loss / n, correct / n

best = 0.0
for epoch in range(15):
    tr_loss, tr_acc = run(train_dl, train=True)
    va_loss, va_acc = run(val_dl)
    print(f"Epoch {epoch+1:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
          f"| val loss {va_loss:.3f} acc {va_acc:.3f}")
    if va_acc > best:
        best = va_acc
        torch.save(model.state_dict(), "best.pt")

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

# ---------- 7. Evaluate best checkpoint on the test set ----------
model.load_state_dict(torch.load("best.pt"))
model.eval()
preds, confs = [], []
with torch.no_grad():
    for xb, _ in test_dl:
        p = torch.softmax(model(xb.to(device)), dim=1).cpu()
        preds += p.argmax(1).tolist()
        confs += p.max(1).values.tolist()

print("\nTest accuracy:", round(sum(int(a == b) for a, b in zip(preds, y_te)) / len(y_te), 4))
print("Macro F1:", round(f1_score(y_te, preds, average="macro"), 4))
print("\n", classification_report(y_te, preds, target_names=labels))
print("Confusion matrix (rows = true, cols = predicted):")
print(labels)
print(confusion_matrix(y_te, preds))

# ---------- 8. Write predictions back to MongoDB ----------
col.update_many({}, {"$unset": {"predicted": "", "confidence": ""}})
ops = [UpdateOne({"_id": id_te[i]},
                 {"$set": {"predicted": labels[preds[i]], "confidence": confs[i]}})
       for i in range(len(preds))]
col.bulk_write(ops)

wrong = list(col.find({"predicted": {"$exists": True},
                       "$expr": {"$ne": ["$category", "$predicted"]}}))
print(f"\nMisclassified: {len(wrong)} of {len(preds)}")
for d in wrong[:5]:
    print(f"true={d['category']} | pred={d['predicted']} | conf={d['confidence']:.2f} | {d['text'][:100]}")