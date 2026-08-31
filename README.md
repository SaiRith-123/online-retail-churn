# 🛒 Online Retail Customer Churn Prediction

An end-to-end **MLOps machine learning project** that predicts customer churn using the **Online Retail II dataset**.

The project implements a complete ML pipeline—from raw data ingestion and automated preprocessing to feature engineering, model training, hyperparameter optimization, experiment tracking, model evaluation, and deployment through an interactive web application with **Explainable AI (XAI)**.

---

## 🚀 Project Overview

Customer churn is a critical business problem in the retail industry. Identifying customers who are likely to stop purchasing allows businesses to take proactive actions such as targeted marketing campaigns, personalized offers, and customer retention strategies.

This project uses historical transaction data to build a machine learning model capable of predicting whether a customer is likely to churn.

The complete workflow includes:

* 📥 Data ingestion
* 🧹 Automated data cleaning
* ⚙️ Feature engineering
* 📊 RFM-style customer behavior analysis
* 🤖 XGBoost model training
* 🔍 Hyperparameter tuning with GridSearchCV
* ⚖️ Class imbalance handling using SMOTE
* 📈 Experiment tracking with MLflow
* 🧪 Model evaluation on a hidden test dataset
* 🧠 Explainable AI predictions
* 🌐 Interactive real-time web application using Gradio
* 📄 Downloadable customer prediction reports

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Raw Retail Data   │
                    │  Online Retail II   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Pipeline     │
                    │   pipeline.py       │
                    └──────────┬──────────┘
                               │
                 Cleaning + Feature Engineering
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Processed Dataset   │
                    │      Parquet        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Model Training     │
                    │      train.py       │
                    └──────────┬──────────┘
                               │
                    XGBoost + SMOTE
                    GridSearchCV
                               │
                               ▼
                    ┌─────────────────────┐
                    │      MLflow         │
                    │ Experiment Tracking │
                    │ Model Registry      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │    evaluate.py      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Gradio Web App    │
                    │     app.py          │
                    │                     │
                    │ • Churn Prediction  │
                    │ • XAI Explanation   │
                    │ • Feature Impact    │
                    │ • Excel Reports     │
                    └─────────────────────┘
```

---

# 📂 Project Structure

```text
online-retail-churn-prediction/
│
├── app.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv
│   │
│   └── processed/
│       └── customer_features.parquet
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   └── train.py
│
├── mlruns/
│
└── artifacts/
    ├── models/
    └── reports/
```

---

# 🔄 Machine Learning Pipeline

## 1️⃣ Data Ingestion & Processing

The pipeline processes **over 1 million retail transactions** from the Online Retail II dataset.

Key preprocessing steps include:

* Removing cancelled transactions
* Removing missing customer IDs
* Handling invalid or incomplete records
* Processing transaction dates
* Preparing clean customer-level datasets

The processed dataset is stored in an efficient **Parquet format** for faster downstream processing.

---

## 2️⃣ Feature Engineering

Customer-level behavioral features are generated using transaction history.

The project uses **RFM-inspired features**:

* **Recency** – How recently a customer made a purchase
* **Frequency** – How often a customer makes purchases
* **Monetary Value** – Total amount spent by the customer

Additional behavioral and transactional features can be derived to improve churn prediction performance.

---

## 3️⃣ Model Training

The project uses an **XGBoost Classifier** for churn prediction.

XGBoost was selected because of its:

* High predictive performance
* Ability to model complex relationships
* Strong performance on structured/tabular data
* Robustness with large datasets

---

## 4️⃣ Handling Class Imbalance

Customer churn datasets are often imbalanced, meaning the number of churned and non-churned customers may differ significantly.

To address this issue, the project uses:

**SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE generates synthetic examples of the minority class, helping the model learn churn patterns more effectively.

---

## 5️⃣ Hyperparameter Optimization

The model is optimized using **GridSearchCV**.

The pipeline evaluates **27 different hyperparameter combinations** to identify the best-performing XGBoost configuration.

The tuning process evaluates combinations of parameters such as:

```text
• Number of estimators
• Learning rate
• Maximum tree depth
```

The best-performing configuration is automatically selected.

---

# 📊 Experiment Tracking with MLflow

MLflow is used to manage and track machine learning experiments.

The pipeline logs:

* Model parameters
* Hyperparameters
* ROC-AUC score
* F1 score
* Training artifacts
* Model versions
* Best-performing model

The final model is registered in the **MLflow Model Registry** for easier model lifecycle management.

This makes experiments reproducible and allows easy comparison between different model versions.

---

# 🧪 Model Evaluation

The trained model is evaluated using a hidden test dataset.

Evaluation metrics include:

* Accuracy
* ROC-AUC Score
* F1 Score
* Confusion Matrix

Run the evaluation using:

```bash
python evaluate.py
```

---

# 🧠 Explainable AI (XAI)

The deployed application does more than simply predict whether a customer will churn.

For every prediction, the application provides:

### 🔍 Dynamic Prediction Explanation

A human-readable explanation describing why the model considers the customer likely or unlikely to churn.

Example:

> "This customer has a high churn risk due to low purchase frequency and a long period since their last transaction."

### 📊 Feature Impact Visualization

A graphical representation showing how different customer features influence the prediction.

### 📄 Downloadable Prediction Report

Users can download prediction results as an Excel report for further analysis.

---

# 🌐 Interactive Web Application

The model is deployed using **Gradio**.

The web application allows users to:

* Adjust customer behavior inputs
* Generate real-time churn predictions
* View churn probability
* Read AI-generated explanations
* Analyze feature impact
* Download prediction reports

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd online-retail-churn-prediction
```

---

## 2️⃣ Create a Virtual Environment

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Setup

Download the **Online Retail II dataset** and place the CSV file inside:

```text
data/raw/
```

The expected structure is:

```text
data/
└── raw/
    └── online_retail_II.csv
```

---

# 🔄 Run the Data Pipeline

Run the automated preprocessing and feature engineering pipeline:

```bash
python src/pipeline.py
```

This process will:

1. Load raw transaction data
2. Remove invalid and cancelled transactions
3. Handle missing values
4. Generate customer-level features
5. Create RFM-style features
6. Save the processed dataset

The output is stored as a processed Parquet file.

---

# 🤖 Train the Model

## macOS / Linux

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m src.train
```

## Windows PowerShell

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python -m src.train
```

The training pipeline performs:

* Data loading
* Train/test splitting
* SMOTE oversampling
* XGBoost training
* GridSearchCV hyperparameter optimization
* MLflow experiment tracking
* Model registration

> ⏱️ **Note:** Training typically takes approximately **3–5 minutes**, depending on your system specifications.

---

# 📈 View the MLflow Dashboard

Launch the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://localhost:5000
```

From the dashboard, you can inspect:

* Experiment runs
* Model parameters
* Performance metrics
* Logged artifacts
* Model versions

---

# 🧪 Evaluate the Model

Run:

```bash
python evaluate.py
```

The evaluation script generates performance metrics on the hidden test dataset, including:

```text
Accuracy
ROC-AUC
F1 Score
Confusion Matrix
```

---

# 🚀 Launch the Web Application

Start the Gradio application:

```bash
python app.py
```

The application will launch on a local URL similar to:

```text
http://127.0.0.1:7860
```

Depending on the configuration, Gradio may also generate a temporary public sharing link.

---

# 🛠️ Technology Stack

| Category              | Technologies             |
| --------------------- | ------------------------ |
| Programming Language  | Python                   |
| Machine Learning      | Scikit-learn             |
| Model                 | XGBoost                  |
| Imbalanced Learning   | SMOTE / imbalanced-learn |
| Experiment Tracking   | MLflow                   |
| Hyperparameter Tuning | GridSearchCV             |
| Data Processing       | Pandas, NumPy            |
| Data Storage          | Parquet                  |
| Web Application       | Gradio                   |
| Explainability        | Feature Impact Analysis  |
| Reporting             | Excel / OpenPyXL         |

---

# 🎯 Key Features

* End-to-end MLOps workflow
* Automated data preprocessing
* Large-scale transaction processing
* RFM-based feature engineering
* XGBoost classification
* Automated hyperparameter tuning
* SMOTE class balancing
* MLflow experiment tracking
* Model registry integration
* Hidden test set evaluation
* Interactive Gradio deployment
* Explainable AI predictions
* Feature impact visualization
* Downloadable Excel reports

---

# 🔮 Future Improvements

Potential improvements for the project include:

* [ ] Docker containerization
* [ ] CI/CD pipeline using GitHub Actions
* [ ] Data and model versioning using DVC
* [ ] Automated model retraining
* [ ] Drift detection and monitoring
* [ ] REST API deployment using FastAPI
* [ ] Cloud deployment
* [ ] Kubernetes deployment
* [ ] SHAP-based explainability
* [ ] Automated data validation
* [ ] Production monitoring dashboards

---

# 👨‍💻 Author

**Sai Rithesh Mandalapu**

AI & Machine Learning Enthusiast | MLOps | DevSecMLOps

---

# ⭐ Support

If you found this project interesting or useful, consider giving the repository a **star ⭐**!

Contributions, suggestions, and feedback are always welcome.
