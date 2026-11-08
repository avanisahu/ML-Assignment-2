import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
sys.path.insert(0, MODEL_DIR)
from utils import MODEL_FILES, TARGET, clean_dataframe, compute_metrics  # noqa: E402

CLASS_NAMES = {0: "No purchase", 1: "Purchase"}

st.set_page_config(page_title="Shopper Intent Classifier", page_icon="🛒", layout="wide")


DATA_PATH = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")


@st.cache_resource(show_spinner="Preparing models (one-time, about a minute)...")
def train_all_models():
    """Fallback: retrain with the same 80/20 split as model/train_models.py.
    Used only when a saved .pkl cannot be read (scikit-learn version mismatch)."""
    from sklearn.model_selection import train_test_split
    from utils import build_models
    d = clean_dataframe(pd.read_csv(DATA_PATH))
    X, y = d.drop(columns=TARGET), d[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    models = build_models()
    for m in models.values():
        m.fit(X_train, y_train)
    return models


@st.cache_resource
def load_model(name):
    try:
        return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))
    except Exception:
        return train_all_models()[name]


st.title("🛍️ Will This Visitor Buy? – Classifier Comparison")
st.write(
    "Upload the **test CSV**, choose a model, and see how well it predicts whether a "
    "website session ends in a purchase (`Revenue`). Dataset: UCI Online Shoppers "
    "Purchasing Intention (17 features, imbalanced target)."
)

st.sidebar.header("⚙️ Your Options")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))
st.sidebar.caption(f"The CSV needs the original feature columns and a `{TARGET}` column.")

sample_path = os.path.join(BASE_DIR, "test_data.csv")
if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test_data.csv", f, "test_data.csv", "text/csv")

if uploaded is None:
    st.info("👈 Upload a CSV in the sidebar to begin (you can download the sample test_data.csv there).")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"The CSV must contain a '{TARGET}' column with the true labels.")
    st.stop()
df = clean_dataframe(df)

model = load_model(model_name)
feature_names = list(model.feature_names_in_)
missing = [c for c in feature_names if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X, y = df[feature_names], df[TARGET]

with st.expander("👀 Preview of uploaded data"):
    st.write(f"Rows: **{len(df)}** | Features: **{len(feature_names)}**")
    st.dataframe(df.head(10), use_container_width=True)

metrics, y_pred = compute_metrics(model, X, y)

st.subheader(f"📊 Evaluation metrics – {model_name}")
for col, (k, v) in zip(st.columns(6), metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
labels = sorted(y.unique())
names = [CLASS_NAMES.get(l, str(l)) for l in labels]
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("🏆 All 5 models on this data")
rows = {n: compute_metrics(load_model(n), X, y)[0] for n in MODEL_FILES}
comp = pd.DataFrame(rows).T.round(4)
comp.index.name = "ML Model Name"
st.dataframe(comp, use_container_width=True)
st.caption(f"Best model by MCC: **{comp['MCC'].idxmax()}**")
