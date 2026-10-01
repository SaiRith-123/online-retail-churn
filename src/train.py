import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.data.ingest import load_config

def train():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_path"])

        # We removed "recency", "tenure_days", and "avg_days_between_purchases" —
    # all three are derived from last_purchase, which correlates almost
    # perfectly with the churn label (recency > 90 days).
    numeric_feats = ["frequency", "monetary", "total_items",
                     "avg_basket_size", "n_unique_products"]
    cat_feats = ["country"]

    X = df[numeric_feats + cat_feats]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], stratify=y, random_state=cfg["model"]["random_state"]
    )

    # 1. Setup the Preprocessing
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_feats),
    ])
    
    # 2. Calculate class imbalance ratio for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 3. Initialize XGBoost
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1 # Set to 1 for GridSearch compatibility
    )
    
    # 4. Build the Pipeline
    pipe = ImbPipeline([
        ("preprocess", pre),
        ("smote", SMOTE(random_state=42)),
        ("clf", xgb)
    ])

    # 5. Define the Grid Search Parameters
    param_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__max_depth': [4, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1]
    }

    # 6. Run GridSearchCV to find the highest accuracy
    print("Starting Grid Search... This will take about 3-5 minutes...")
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_

    # 7. MLflow Tracking
    mlflow.set_experiment("online_retail_churn")

    with mlflow.start_run(run_name="xgboost_tuned") as run:
        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
        }

        # Log the best parameters found by the Grid Search
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            best_model, 
            artifact_path="model",
            skops_trusted_types=[
                'imblearn.over_sampling._smote.base.SMOTE', 
                'imblearn.pipeline.Pipeline', 
                'scipy.sparse._csr.csr_matrix', 
                'xgboost.core.Booster', 
                'xgboost.sklearn.XGBClassifier'
            ]
        )

        print("\n=== Training Complete! ===")
        print(f"Best Parameters Found: {grid.best_params_}")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        # Register the best model
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="churn_classifier_prod"
        )

if __name__ == "__main__":
    train()