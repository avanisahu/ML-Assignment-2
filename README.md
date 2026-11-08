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

https://github.com/avanisahu/ML-Assignment-2

Live Streamlit app: https://ml-assignment-2-hhwfjbnn65tcud3br2nmu8.streamlit.app/

```
project-folder/

│-- data/online_shoppers_intention.csv
│-- model/ (train_models.py, utils.py, make_readme.py, *.pkl)
|-- .gitignore
│-- README.md
│-- app.py
│-- requirements.txt
│-- test_data.csv
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
| Logistic Regression | Accuracy 85.0%, AUC 0.896, MCC 0.534. Class weighting lets it find about 75% of buyers (recall 0.749), but precision is only 0.51, so roughly half of the predicted buyers do not buy. A solid, simple baseline. |
| Decision Tree | Best overall: MCC 0.564, AUC 0.905 and the highest recall (0.835). Splits on features like PageValues suit this data, and the depth limit of 6 keeps it from overfitting. The cost is low precision (0.50). |
| kNN | Highest accuracy (87.4%) and precision (0.67) but recall of only 0.36, so it misses about 64% of buyers. The majority "no purchase" class dominates the neighbours, which makes accuracy misleading here (MCC 0.432). |
| Naive Bayes | Weakest model: accuracy 67.4%, MCC 0.325. It flags many sessions as buyers (recall 0.79, precision 0.29) because correlated features such as bounce and exit rates break its independence assumption. |
| Support Vector Machine | Very close to Logistic Regression (accuracy 85.7%, MCC 0.532, AUC 0.884). The RBF kernel gives no clear gain over the linear model here but takes longer to train. |
| **Overall winner** | **Decision Tree** – it has the highest MCC (0.564), the highest AUC (0.905) and the best recall. MCC is the deciding metric because only about 15% of sessions are purchases, so accuracy alone favours models like kNN that miss most buyers. |
