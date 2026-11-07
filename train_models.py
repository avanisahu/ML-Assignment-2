"""Train the 5 classifiers on the Online Shoppers Purchasing Intention dataset.

1. Download the dataset from
   https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
   and put  online_shoppers_intention.csv  inside the  data/  folder.
2. From the project root run:   python model/train_models.py

It saves the models, test_data.csv, metrics_summary.csv and writes the
results section of README.md for you.
"""
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (build_models, compute_metrics, clean_dataframe,
                   MODEL_FILES, TARGET)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(ROOT, "data", "online_shoppers_intention.csv"))

if not os.path.exists(DATA_PATH):
    sys.exit(f"Dataset not found at {DATA_PATH}\n"
             "Download online_shoppers_intention.csv from the UCI link in this file's "
             "docstring and place it in the data/ folder.")

# 1. Load + clean
df = clean_dataframe(pd.read_csv(DATA_PATH))
X, y = df.drop(columns=TARGET), df[TARGET]

# 2. Stratified 80/20 split (important: only ~15% of sessions end in a purchase)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

# 3. Train, evaluate, save
rows = {}
for name, mdl in build_models().items():
    print("Training", name, "...", flush=True)
    mdl.fit(X_train, y_train)
    rows[name] = compute_metrics(mdl, X_test, y_test)[0]
    joblib.dump(mdl, os.path.join(MODEL_DIR, MODEL_FILES[name]))

res = pd.DataFrame(rows).T
res.index.name = "ML Model Name"
res.round(4).to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"))
print(res.round(4).to_string())

# 4. Write README.md from template with the real numbers
import make_readme  # noqa: E402
make_readme.write_readme(ROOT, df, X, y, X_train, X_test, res)
print("\nDone. README.md, test_data.csv and model/*.pkl are ready.")
