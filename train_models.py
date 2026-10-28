"""Train all five classifiers and save models plus a held-out test CSV."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "bank-full.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "bank-full.csv not found. Download the UCI Bank Marketing ZIP, "
        "extract bank-full.csv, and place it beside train_models.py."
    )

# The UCI bank-full.csv is semicolon-separated, with 16 input features and y.
df = pd.read_csv(DATA_PATH, sep=";")
if "y" not in df.columns:
    raise ValueError("Expected target column 'y' in bank-full.csv.")

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})
if y.isna().any():
    raise ValueError("Target column y must contain only 'no' and 'yes'.")

# Save a reproducible held-out test set, including target column.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
test_df = X_test.copy()
test_df["y"] = y_test.map({0: "no", 1: "yes"})
test_df.to_csv(ROOT / "test_data.csv", index=False)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

# Dense output is required by GaussianNB. The bank-full data is small enough
# for this representation on a typical laptop/Colab runtime.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
}

results = []
for name, estimator in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", estimator)])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))

    pred = pipeline.predict(X_test)
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(ROOT / "training_metrics.csv", index=False)
print("\nEvaluation on the held-out test split:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved models in: {MODEL_DIR}")
print(f"Saved test data: {ROOT / 'test_data.csv'}")
