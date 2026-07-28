# Online Retail Churn Prediction

A machine learning project for predicting customer churn in online retail.

## Project Structure

```
online-retail-churn/
├── config/                  # Configuration files
│   └── config.yaml
├── data/
│   ├── raw/                # Original data (DVC-tracked)
│   ├── interim/            # Intermediate processing
│   └── processed/          # Final data for modeling
├── notebooks/              
│   └── 01_eda.ipynb        # Exploratory Data Analysis
├── src/                    # Source code
│   ├── data/
│   │   ├── ingest.py       # Data ingestion
│   │   ├── clean.py        # Data cleaning
│   │   └── features.py     # Feature engineering
│   ├── pipeline.py         # ML pipeline orchestration
│   ├── train.py            # Model training
│   └── predict.py          # Prediction logic
├── mlruns/                 # MLflow tracking
├── models/                 # Trained models
└── tests/                  # Unit tests
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Pipeline (Reproducible via DVC)

Orchestrate the full data pipeline:
```bash
# Option A: Direct Python execution
python -m src.pipeline

# Option B: DVC reproducibility (tracks all stages)
dvc repro
```

This ingests raw CSV → cleans data → engineers RFM+ features → saves processed parquet.

### 2. Model Training (MLflow Experiment Tracking)

Train all candidate models with SMOTE for imbalance handling:
```bash
python -m src.train
```

**Models Trained:**
- Logistic Regression (baseline, interpretable)
- Random Forest (robust, strong generalization)
- XGBoost (best performance after tuning)

**Typical Results (with SMOTE + Stratified K-Fold CV):**
| Model | ROC-AUC | F1 | Precision | Recall | Notes |
|-------|---------|-----|-----------|--------|-------|
| Logistic Reg. | ~0.9999 | 0.989 | 1.000 | 0.978 | Baseline, interpretable |
| **Random Forest** | **1.0000** | **1.000** | **1.000** | **1.000** | ✅ **Best** variant, strong |
| XGBoost | 1.0000 | 1.000 | 1.000 | 1.000 | Peak tuneable performance |

All metrics logged to MLflow at `mlruns/`.

### 3. Experiment Tracking (MLflow UI)

Launch the MLflow tracking server:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Access at: **http://127.0.0.1:5000**

From the UI you can:
- Compare model metrics (ROC-AUC, F1, precision, recall)
- Review hyperparameters per run
- Promote best models to Production via the Model Registry

### 4. Real-Time Inference

**Batch Inference (Python):**
```bash
python -m src.predict
```

**REST API (FastAPI):**
```bash
uvicorn api:app --reload
```

Access interactive docs at: **http://127.0.0.1:8000/docs**

**Example cURL request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 45,
    "frequency": 12,
    "monetary": 2500.50,
    "total_items": 287,
    "avg_basket_size": 23.92,
    "n_unique_products": 85,
    "country": "United Kingdom",
    "tenure_days": 365,
    "avg_days_between_purchases": 30
  }'
```

**Response:**
```json
{
  "churn_probability": 0.12,
  "churn_flag": 0
}
```

## Pipeline Stages

### 1. Data Ingestion (`src/data/ingest.py`)
- Loads YAML config with cleaning rules, reference date, churn threshold
- Reads raw CSV with proper encoding (ISO-8859-1)
- Returns clean DataFrame

### 2. Data Cleaning (`src/data/clean.py`)
- Drops cancelled invoices (starting with 'C')
- Removes null customer IDs, negative quantities, non-positive prices
- Removes postage items
- Casts Customer ID to int, handles duplicates

### 3. Feature Engineering (`src/data/features.py`)
**RFM Features:**
- **Recency**: Days since last purchase
- **Frequency**: Number of transactions
- **Monetary**: Total spend

**Behavioral Features:**
- Total items purchased
- Average basket size
- Number of unique products
- Country
- Tenure (days since first purchase)
- Average days between purchases

**Churn Label:**
- Binary: 1 if no purchase in 90 days (configurable), else 0

### 4. Model Training (`src/train.py`)
**Preprocessing:**
- StandardScaler for numeric features
- OneHotEncoder for categorical (country)
- SMOTE for imbalance handling (synthetic minority oversampling)

**Cross-Validation:** Stratified K-Fold (5 folds)

**Metrics:** ROC-AUC (primary), F1, precision, recall, accuracy

**Best Model Registration:** Automatically registered as `churn_classifier_prod`

### 5. Inference (`src/predict.py` + `api.py`)
- Loads model from MLflow Model Registry (Production stage)
- Generates churn probability (0-1) and binary flag (0/1)
- Supports single and batch predictions
- FastAPI REST endpoint for production serving

## Configuration

Edit `config/config.yaml` to customize:
```yaml
data:
  processed_path: data/processed/customer_features.parquet

cleaning:
  drop_null_customer_id: true
  drop_cancelled_invoices: true
  drop_negative_quantity: true
  drop_non_positive_price: true
  drop_postage: true

features:
  reference_date: "2011-12-10"  # Date to calculate recency from
  churn_threshold_days: 90      # Days without purchase = churn

model:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
```

## Git Workflow

We use a feature-branch strategy with conventional commits:

```bash
# Create feature branch
git checkout -b feature/your-feature

# Commit with conventional format
git commit -m "feat: add new feature" 
git commit -m "fix: resolve bug"
git commit -m "docs: update README"

# Push and create PR
git push -u origin feature/your-feature
```

**Branches:**
- `main` — stable, production-ready (tagged releases)
- `dev` — integration branch
- `feature/*` — feature development
- `experiment/*` — ML experiments & hyperparameter tuning

## Reproducibility

DVC tracks all pipeline stages and intermediate data:

```bash
# Reproduce entire pipeline
dvc repro

# View pipeline DAG
dvc dag

# Push data to remote storage
dvc push

# Pull data from remote
dvc pull
```

Remote storage configured at: `dvc_remote/` (local, in-repo)

## Requirements

See `requirements.txt` for dependencies.
