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

Run the complete pipeline:
```bash
python src/pipeline.py
```

## Requirements

See `requirements.txt` for dependencies.
