"""
Model Building script with experimentation tracking (MLflow).

- Loads the train/test splits produced by prep.py.
- Builds a preprocessing + model pipeline (one-hot encodes the categorical
  columns so the Streamlit app can submit raw string values at inference).
- Tunes several candidate algorithms with GridSearchCV.
- Logs every run's parameters and metrics to MLflow.
- Evaluates each candidate and selects the best by ROC-AUC.
- Saves the best model into tourism_project/deployment/ so the pipeline
  can commit it alongside the Streamlit app.
Run from the repository root: python tourism_project/model_building/train.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import mlflow

DATA_DIR = "tourism_project/data"
DEPLOY_DIR = "tourism_project/deployment"
TARGET_COL = "ProdTaken"

CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


def make_preprocessor():
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")

    return metrics


CANDIDATES = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [5, 10, 15],
         "model__min_samples_split": [2, 5, 10],
         "model__min_samples_leaf": [1, 2, 4]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [10, 15, None],
         "model__min_samples_split": [2, 5],
         "model__min_samples_leaf": [1, 2]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1, 0.15],
         "model__max_depth": [3, 5, 7]},
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1, 0.15],
         "model__max_depth": [3, 5, 7],
         "model__subsample": [0.8, 0.9]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"model__n_estimators": [50, 100, 200],
         "model__learning_rate": [0.5, 1.0, 1.5]},
    ),
}


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    mlflow.set_experiment("tourism_package_prediction")

    models_results = []

    for name, (estimator, param_grid) in CANDIDATES.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ("preprocess", make_preprocessor()),
                ("model", estimator),
            ])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("model_type", name)

            metrics = evaluate_model(best_pipeline, X_test, y_test, name)
            mlflow.log_metrics(metrics)

            models_results.append((name, best_pipeline, metrics["roc_auc"]))

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    results_df = pd.DataFrame(
        [(n, s) for n, _, s in models_results], columns=["Model", "ROC_AUC"]
    ).sort_values("ROC_AUC", ascending=False)
    print(results_df.to_string(index=False))

    best_model_name, best_model, best_score = max(models_results, key=lambda x: x[2])
    print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_score:.4f})")

    os.makedirs(DEPLOY_DIR, exist_ok=True)
    model_path = os.path.join(DEPLOY_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    results_df.to_csv("tourism_project/model_building/model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
