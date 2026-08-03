# 🛒 Online Retail Customer Churn Prediction

An end-to-end MLOps project to predict customer churn using the Online Retail II dataset. The pipeline covers data ingestion, cleaning, feature engineering, model training with MLflow tracking, and real-time inference with FastAPI.

## 🏗️ Project Architecture

- **Data Pipeline**: Cleans raw transactional data and engineers RFM-style customer features.
- **Model Training**: Trains a Random Forest classifier with SMOTE to handle class imbalance.
- **Experiment Tracking**: MLflow logs model metrics, parameters, and artifacts.
- **Model Registry**: The best run is registered to the MLflow model registry under the `Production` stage.
- **Real-Time API**: FastAPI exposes a prediction endpoint for live churn scoring.

## 🚀 How to Run This Project

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# OR
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

### 2. Place the Dataset

Put `online_retail_II.csv` in the `data/raw/` folder before running the pipeline.

### 3. Run the Data Pipeline

```bash
python src/pipeline.py
```

This step ingests the raw CSV, cleans the transactions, engineers customer features, and writes the processed features to parquet.

### 4. Train the Model and Log to MLflow

On macOS/Linux:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python src/train.py
```

On Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python src/train.py
```

### 5. View the MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000/) to review experiments and metrics.

### 6. Serve Real-Time Predictions

```bash
uvicorn api:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to test the API interactively.

---

## ✅ Final Validation

### 1. Run the Tests

```bash
pytest tests/
```

You should see a passing pytest result for the repository test suite.

### 2. Start the API

```bash
uvicorn api:app --reload
```

### 3. Test the API

Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), open `POST /predict`, then click **Try it out** and submit the sample payload below:

```json
{
  "frequency": 2,
  "monetary": 50.0,
  "total_items": 5,
  "avg_basket_size": 25.0,
  "n_unique_products": 3,
  "country": "United Kingdom",
  "tenure_days": 200,
  "avg_days_between_purchases": 100.0
}
```

The response will include a churn probability and predicted churn flag.

## 🧪 Save Everything to Git

```bash
git add .
git commit -m "feat: finalize end-to-end churn project with API and tests"
git push
```
