"""Generates README.md (assignment Step 5 structure) from the real results."""
import os

NOTES = {
    "Logistic Regression": "A linear baseline on scaled + one-hot features; class weighting is used because only about 15% of sessions end in a purchase. Fast and interpretable, but cannot capture non-linear interactions between page values, exit rates and month.",
    "Decision Tree": "A depth-limited tree (max_depth=6) that can model non-linear rules such as high PageValues leading to purchase. Easy to read, but a single tree can overfit or be unstable on noisy browsing data.",
    "kNN": "Distance based (k=7) on standardised features. It has no class weighting, so it tends to favour the majority 'no purchase' class, which usually lowers recall on the minority class.",
    "Naive Bayes": "Gaussian Naive Bayes assumes features are independent and normally distributed. Duration, count and rate features are correlated and skewed, so its probabilities are less reliable, though it is very fast.",
    "Support Vector Machine": "RBF-kernel SVM with class weighting. It can learn a flexible non-linear boundary, but it is the slowest to train and the hardest to interpret.",
}


def _md_table(df):
    cols = list(df.columns)
    out = ["| ML Model Name | " + " | ".join(cols) + " |",
           "|---|" + "---|" * len(cols)]
    for name, r in df.iterrows():
        out.append(f"| {name} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
    return "\n".join(out)


def write_readme(root, df, X, y, X_train, X_test, res):
    r = res.round(4)
    best_mcc, best_auc = r["MCC"].idxmax(), r["AUC"].idxmax()
    pos = int(y.sum()); neg = len(y) - pos
    obs_rows = []
    for name in r.index:
        m = r.loc[name]
        rank = int(r["MCC"].rank(ascending=False)[name])
        obs_rows.append(
            f"| {name} | MCC {m['MCC']:.4f} (rank {rank} of 5), AUC {m['AUC']:.4f}, "
            f"accuracy {m['Accuracy']:.4f}, precision {m['Precision']:.4f}, recall {m['Recall']:.4f}. "
            f"{NOTES[name]} |")
    winner = (f"**{best_mcc}** has the best MCC ({r.loc[best_mcc, 'MCC']:.4f}) and "
              f"{best_auc} has the best AUC ({r.loc[best_auc, 'AUC']:.4f}). "
              "MCC is the main criterion because the classes are imbalanced and accuracy alone is misleading.")

    text = f"""# Online Shoppers Purchase Prediction – Classification Models & Streamlit App

Machine Learning – Assignment 2 (M.Tech DSE, BITS Pilani WILP)

## a. Problem Statement

Predict whether an e-commerce website visit (session) ends in a **purchase** (`Revenue = 1`) or not (`Revenue = 0`) from the visitor's browsing behaviour, page metrics and session details. This is a **binary classification** problem with imbalanced classes. Five classifiers are trained on the same data, compared with six metrics, and demonstrated in a Streamlit web app.

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Instances | {len(df)} sessions (minimum required: 500) |
| Features | {X.shape[1]} input features (minimum required: 12) – 10 numeric browsing metrics (Administrative, Informational, ProductRelated pages and durations, BounceRates, ExitRates, PageValues, SpecialDay) plus OperatingSystems, Browser, Region, TrafficType, Weekend, Month and VisitorType |
| Target | `Revenue` – 1 = purchase ({pos} sessions), 0 = no purchase ({neg} sessions) |
| Missing values | {int(df.isna().sum().sum())} |

**Pre-processing:** numeric columns are standardised, `Month` and `VisitorType` are one-hot encoded, `Weekend`/`Revenue` are converted to 0/1. All of this is inside a scikit-learn `Pipeline`, so the app accepts the raw CSV columns.

**Split:** 80% train ({len(X_train)} rows) / 20% test ({len(X_test)} rows), stratified, `random_state=42`. The test rows are saved in `test_data.csv`. Precision, Recall and F1 are for the positive class (purchase).

## c. GitHub Repository Link

<PASTE YOUR GITHUB REPO LINK HERE>

Live Streamlit app: <PASTE YOUR STREAMLIT APP LINK HERE>

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- data/online_shoppers_intention.csv
│-- model/ (train_models.py, utils.py, make_readme.py, *.pkl, metrics_summary.csv)
```

Run locally: `pip install -r requirements.txt`, then `python model/train_models.py`, then `streamlit run app.py`.

## d. Models Used

{_md_table(r)}

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
{chr(10).join(obs_rows)}
| **Overall winner for this dataset** | {winner} |
"""
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(text)
