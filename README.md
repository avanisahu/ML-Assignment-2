# Online Shoppers Purchase Prediction – Classification Models & Streamlit App

Machine Learning – Assignment 2 (M.Tech DSE, BITS Pilani WILP)

## a. Problem Statement

Predict whether an e-commerce website visit (session) ends in a **purchase** (`Revenue = 1`) or not (`Revenue = 0`) from the visitor's browsing behaviour, page metrics and session details. This is a **binary classification** problem with imbalanced classes. Five classifiers are trained on the same data, compared with six metrics, and demonstrated in a Streamlit web app.

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Instances | 12330 sessions (minimum required: 500) |
| Features | 17 input features (minimum required: 12) – 10 numeric browsing metrics (Administrative, Informational, ProductRelated pages and durations, BounceRates, ExitRates, PageValues, SpecialDay) plus OperatingSystems, Browser, Region, TrafficType, Weekend, Month and VisitorType |
| Target | `Revenue` – 1 = purchase (1908 sessions), 0 = no purchase (10422 sessions) |
| Missing values | 0 |

**Pre-processing:** numeric columns are standardised, `Month` and `VisitorType` are one-hot encoded, `Weekend`/`Revenue` are converted to 0/1. All of this is inside a scikit-learn `Pipeline`, so the app accepts the raw CSV columns.

**Split:** 80% train (9864 rows) / 20% test (2466 rows), stratified, `random_state=42`. The test rows are saved in `test_data.csv`. Precision, Recall and F1 are for the positive class (purchase).

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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8504 | 0.8962 | 0.5116 | 0.7487 | 0.6079 | 0.5338 |
| Decision Tree | 0.8455 | 0.9046 | 0.5008 | 0.8351 | 0.6261 | 0.5642 |
| kNN | 0.8739 | 0.7993 | 0.6715 | 0.3639 | 0.4720 | 0.4322 |
| Naive Bayes | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Support Vector Machine | 0.8569 | 0.8840 | 0.5279 | 0.7173 | 0.6082 | 0.5323 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | MCC 0.5338 (rank 2 of 5), AUC 0.8962, accuracy 0.8504, precision 0.5116, recall 0.7487. A linear baseline on scaled + one-hot features; class weighting is used because only about 15% of sessions end in a purchase. Fast and interpretable, but cannot capture non-linear interactions between page values, exit rates and month. |
| Decision Tree | MCC 0.5642 (rank 1 of 5), AUC 0.9046, accuracy 0.8455, precision 0.5008, recall 0.8351. A depth-limited tree (max_depth=6) that can model non-linear rules such as high PageValues leading to purchase. Easy to read, but a single tree can overfit or be unstable on noisy browsing data. |
| kNN | MCC 0.4322 (rank 4 of 5), AUC 0.7993, accuracy 0.8739, precision 0.6715, recall 0.3639. Distance based (k=7) on standardised features. It has no class weighting, so it tends to favour the majority 'no purchase' class, which usually lowers recall on the minority class. |
| Naive Bayes | MCC 0.3249 (rank 5 of 5), AUC 0.7939, accuracy 0.6736, precision 0.2941, recall 0.7906. Gaussian Naive Bayes assumes features are independent and normally distributed. Duration, count and rate features are correlated and skewed, so its probabilities are less reliable, though it is very fast. |
| Support Vector Machine | MCC 0.5323 (rank 3 of 5), AUC 0.8840, accuracy 0.8569, precision 0.5279, recall 0.7173. RBF-kernel SVM with class weighting. It can learn a flexible non-linear boundary, but it is the slowest to train and the hardest to interpret. |
| **Overall winner for this dataset** | **Decision Tree** has the best MCC (0.5642) and Decision Tree has the best AUC (0.9046). MCC is the main criterion because the classes are imbalanced and accuracy alone is misleading. |
