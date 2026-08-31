🛒 Online Retail Customer Churn Prediction
An end-to-end MLOps project to predict customer churn using the Online Retail II dataset. The pipeline covers data ingestion, automated cleaning, feature engineering, model training with MLflow tracking, hyperparameter tuning, and a real-time interactive web app with Explainable AI (XAI).

🏗️ Project Architecture
Automated Data Pipeline (src/pipeline.py): Ingests 1M+ raw transactions, cleans data (removing cancellations and nulls), and engineers RFM-style customer features.
Model Training & Tuning (src/train.py): Trains a highly accurate XGBoost classifier. Uses GridSearchCV to test 27 hyperparameter combinations and SMOTE to handle class imbalance.
Experiment Tracking: MLflow logs all model metrics (ROC-AUC, F1), parameters, and artifacts. The best model is registered to the MLflow Model Registry.
Explainable AI Web App (app.py): A Gradio web application that not only predicts churn probability but generates a dynamic text explanation, displays a Feature Impact Graph, and generates a downloadable Excel report.
🚀 How to Run This Project
1. Setup Environment
bash

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# OR
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
2. Place the Dataset
Put online_retail_II.csv in the data/raw/ folder before running the pipeline.

3. Run the Data Pipeline
bash

python src/pipeline.py
This ingests the raw CSV, cleans the transactions, engineers customer features, and saves a processed parquet file.

4. Train the Model and Log to MLflow
On macOS/Linux:

bash

export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m src.train
On Windows PowerShell:

powershell

 $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python -m src.train
(Note: This takes 3-5 minutes as it runs GridSearchCV to find the optimal XGBoost parameters).

5. View the MLflow Dashboard
bash

mlflow ui --backend-store-uri sqlite:///mlflow.db
Open http://localhost:5000 to review experiments, metrics, and the registered model.

6. Evaluate the Model
Check the final accuracy, ROC-AUC score, and Confusion Matrix on the hidden test set:

bash

python evaluate.py
7. Launch the Real-Time Web App
bash

python app.py
Open the local URL (usually http://127.0.0.1:7860) or the public gradio.live link provided in the terminal to interact with the predictive UI. Adjust the sliders to see real-time predictions, view the explanation, and download the Excel report!