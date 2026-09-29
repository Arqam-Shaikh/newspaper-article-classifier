# Newspaper Article Categorization

Classifies news articles into 5 categories (business, entertainment, politics,
sport, tech) using PyTorch and scikit-learn, with articles and predictions stored in MongoDB.

## Pipeline
MongoDB -> stratified split (70/15/15) -> tokenize / TF-IDF -> model -> evaluation
-> predictions written back to MongoDB -> error analysis

## Results (319-article test set)
| Model | Accuracy | Macro F1 |
|---|---|---|
| BiLSTM (PyTorch) | 0.9216 | 0.919 |
| TF-IDF + Logistic Regression | 0.9843 | 0.9849 |

## Key findings
- The BiLSTM overfit: training accuracy reached ~99% while validation stayed near 90%.
- The linear baseline beat the LSTM on this small dataset (~1,500 training articles),
  because topic labels are largely keyword-driven.
- The first version failed on short sentences because it was trained only on full
  articles. Adding random 10-40 word snippets from the training articles fixed this
  (short-snippet accuracy: X before, Y after).

## Setup
```bash
pip install -r requirements.txt
python get_data.py      # downloads the BBC dataset to bbc-text.csv
python load_data.py     # loads it into a local MongoDB
python train.py         # trains the BiLSTM (saves best.pt, vocab.json)
python baseline.py      # trains the TF-IDF model (saves tfidf_model.joblib)
```
Requires a local MongoDB running at `mongodb://localhost:27017`.

## Classify your own text
```bash
python predict.py "paste an article or sentence here"
python predict.py --file article.txt
python predict.py            # interactive mode
```
`predict.py` uses the TF-IDF model. The BiLSTM version is kept as `predict_lstm.py`
for comparison; it is less reliable on short text.

## Limitations
The models learned from BBC news of roughly 2004-2005 and know only these 5
categories, so unrelated or very modern topics may be misclassified.