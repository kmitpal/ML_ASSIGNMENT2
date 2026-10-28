from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")
st.title("🏦 Bank Marketing Subscription Classifier")
st.write(
    "Compare five saved machine-learning classifiers on a labeled Bank Marketing "
    "test CSV. The app loads saved models; it does not retrain them on uploaded data."
)

st.info(
    "Upload a CSV with the 16 original input columns and the target column `y` "
    "(`yes`/`no`). You can start with the repository's `test_data.csv`."
)

missing_models = [name for name, file in MODEL_FILES.items() if not (MODEL_DIR / file).exists()]
if missing_models:
    st.error("Saved models are missing. Run `python train_models.py` first. Missing: " + ", ".join(missing_models))
    st.stop()

model_name = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    default_path = ROOT / "test_data.csv"
    if default_path.exists():
        st.caption("No file uploaded. Using the included test_data.csv.")
        df = pd.read_csv(default_path)
    else:
        st.warning("Upload a labeled test CSV to continue.")
        st.stop()
else:
    df = pd.read_csv(uploaded, sep=None, engine="python")

if "y" not in df.columns:
    st.error("The CSV must include the target column `y` with values `yes` or `no`.")
    st.stop()

y_raw = df["y"].astype(str).str.strip().str.lower()
y_true = y_raw.map({"no": 0, "yes": 1})
if y_true.isna().any():
    st.error("Target `y` must contain only `yes` and `no`.")
    st.stop()

X = df.drop(columns=["y"])
model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])

expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(model.named_steps["preprocess"].feature_names_in_)
missing = [c for c in expected if c not in X.columns]
if missing:
    st.error("Missing required input columns: " + ", ".join(missing))
    st.stop()
X = X[expected]

pred = model.predict(X)
clf = model.named_steps["classifier"]
scores = model.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else model.decision_function(X)

metrics = {
    "Accuracy": accuracy_score(y_true, pred),
    "AUC": roc_auc_score(y_true, scores) if y_true.nunique() == 2 else float("nan"),
    "Precision": precision_score(y_true, pred, zero_division=0),
    "Recall": recall_score(y_true, pred, zero_division=0),
    "F1 Score": f1_score(y_true, pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, pred),
}
st.subheader(f"Evaluation results — {model_name}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted no", "Predicted yes"],
                yticklabels=["Actual no", "Actual yes"], ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_true, pred, target_names=["no", "yes"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

out = X.copy()
out["actual_y"] = y_raw.values
out["predicted_y"] = ["yes" if p == 1 else "no" for p in pred]
out["model_score"] = scores
st.subheader("Predictions")
st.dataframe(out.head(100), use_container_width=True)
st.download_button(
    "Download predictions as CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="bank_marketing_predictions.csv",
    mime="text/csv",
)
