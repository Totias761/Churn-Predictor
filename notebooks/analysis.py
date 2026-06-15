import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

print(" Imports done")

# --- Load data ---
df = pd.read_csv("data/raw/github_features.csv")
X = df.drop(columns=["username", "churned"])
y = df["churned"]
print(f"Shape: {X.shape} | Churn rate: {y.mean():.1%}")

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- METHOD 1: Filter ---
print("\n" + "="*50)
print("METHOD 1 — FILTER METHODS")
print("="*50)
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_scaled, y)
filter_scores = pd.DataFrame({
    "feature": X.columns,
    "f_score": selector.scores_,
}).sort_values("f_score", ascending=False).reset_index(drop=True)
filter_scores["filter_rank"] = filter_scores.index + 1
print(filter_scores.to_string(index=False))

# --- METHOD 2: RFE ---
print("\n" + "="*50)
print("METHOD 2 — WRAPPER: RFE")
print("="*50)
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5)
rfe.fit(X_scaled, y)
rfe_df = pd.DataFrame({
    "feature": X.columns,
    "rfe_selected": ["" if s else "" for s in rfe.support_],
    "ranking": rfe.ranking_
}).sort_values("ranking")
print(rfe_df.to_string(index=False))

# --- METHOD 3: Decision Tree ---
print("\n" + "="*50)
print("METHOD 3 — DECISION TREE")
print("="*50)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_scaled, y)
dt_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": dt.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
dt_importance["dt_rank"] = dt_importance.index + 1
print(dt_importance.to_string(index=False))

# --- METHOD 4: Random Forest ---
print("\n" + "="*50)
print("METHOD 4 — RANDOM FOREST")
print("="*50)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
rf_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
rf_importance["rf_rank"] = rf_importance.index + 1
print(rf_importance.to_string(index=False))

# --- Comparison Table ---
print("\n" + "="*50)
print("FEATURE SELECTION COMPARISON TABLE")
print("="*50)
comparison = filter_scores[["feature", "filter_rank"]] \
    .merge(rfe_df[["feature", "rfe_selected"]], on="feature") \
    .merge(dt_importance[["feature", "dt_rank"]], on="feature") \
    .merge(rf_importance[["feature", "rf_rank"]], on="feature") \
    .sort_values("rf_rank")
print(comparison.to_string(index=False))
comparison.to_csv("data/raw/feature_comparison.csv", index=False)
print("\n Comparison table saved!")

# --- Cross Validation ---
print("\n" + "="*50)
print("FINAL MODEL — CROSS VALIDATION")
print("="*50)
top_features = rf_importance.head(5)["feature"].tolist()
print(f"Top 5 features from RF: {top_features}")

X_final = df[top_features]
X_final_scaled = scaler.fit_transform(X_final)

final_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
cv_results = cross_validate(
    final_model, X_final_scaled, y,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"]
)
print(f"Accuracy:  {cv_results['test_accuracy'].mean():.3f}")
print(f"Precision: {cv_results['test_precision'].mean():.3f}")
print(f"Recall:    {cv_results['test_recall'].mean():.3f}")
print(f"F1 Score:  {cv_results['test_f1'].mean():.3f}")

# --- Save model ---
final_model.fit(X_final_scaled, y)
joblib.dump(final_model, "app/model.pkl")
joblib.dump(scaler, "app/scaler.pkl")
joblib.dump(top_features, "app/features_list.pkl")
print("\n Model saved to app/model.pkl")
print(" Scaler saved to app/scaler.pkl")
print(" Features list saved to app/features_list.pkl")