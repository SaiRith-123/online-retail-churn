import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.ingest import load_config


NUMERIC_FEATURES = [
    "frequency",
    "monetary",
    "total_items",
    "avg_basket_size",
    "n_unique_products",
    "tenure_days",
    "avg_days_between_purchases",
]
CATEGORICAL_FEATURES = ["country"]


def build_model_pipeline() -> ImbPipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return ImbPipeline(
        [
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("clf", classifier),
        ]
    )


def export_model() -> str:
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_path"])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]

    pipe = build_model_pipeline()
    pipe.fit(X, y)

    output_path = "model.joblib"
    joblib.dump(pipe, output_path)
    print(f"✅ Model exported successfully to {output_path}")
    return output_path


if __name__ == "__main__":
    export_model()
