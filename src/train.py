import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn as mlflow_sklearn
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.data.ingest import load_config


def load_features(cfg):
    return pd.read_parquet(cfg["data"]["processed_path"])


def build_model_pipeline(model, numeric_feats, cat_feats):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_feats),
    ])
    return ImbPipeline([
        ("preprocess", pre),
        ("smote", SMOTE(random_state=42)),
        ("clf", model),
    ])


def evaluate(y_true, y_pred, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def train():
    cfg = load_config()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    df = load_features(cfg)

        # Removed "recency" to prevent data leakage!
    numeric_feats = ["frequency", "monetary", "total_items",
                     "avg_basket_size", "n_unique_products", "tenure_days",
                     "avg_days_between_purchases"]
    cat_feats = ["country"]

    X = df[numeric_feats + cat_feats]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        stratify=y, random_state=cfg["model"]["random_state"]
    )

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            eval_metric="logloss", random_state=42, n_jobs=-1),
    }

    mlflow.set_experiment("online_retail_churn")
    best_run = None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_model_pipeline(model, numeric_feats, cat_feats)
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_params(model.get_params())
            mlflow.log_param("model_name", name)
            mlflow.log_metrics(metrics)
            mlflow_sklearn.log_model(
                pipe,
                name="model",
                skops_trusted_types=[
                    "imblearn.over_sampling._smote.base.SMOTE",
                    "imblearn.pipeline.Pipeline",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ]
            )

            cv = StratifiedKFold(n_splits=cfg["model"]["cv_folds"],
                                 shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train,
                                        cv=cv, scoring="roc_auc", n_jobs=-1)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())

            print(f"\n=== {name} ===")
            print(classification_report(y_test, y_pred))
            print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

            if best_run is None or metrics["roc_auc"] > best_run["roc_auc"]:
                best_run = {"name": name, "run_id": run.info.run_id, **metrics}

    if best_run is None:
        raise RuntimeError("Training did not produce a valid model run.")

    print(f"\n✅ Best model: {best_run['name']} (AUC={best_run['roc_auc']:.4f})")

    # Register best model and promote it to Production
    model_details = mlflow.register_model(
        model_uri=f"runs:/{best_run['run_id']}/model",
        name="churn_classifier_prod"
    )

    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name="churn_classifier_prod",
        version=model_details.version,
        stage="Production",
        archive_existing_versions=True,
    )


if __name__ == "__main__":
    train()
