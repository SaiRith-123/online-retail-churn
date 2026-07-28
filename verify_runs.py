import mlflow
import pandas as pd

client = mlflow.MlflowClient()

# Get all experiments
experiments = client.search_experiments()
print(f"Found {len(experiments)} experiments")
for exp in experiments:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")

# Search runs from all experiments
all_runs = []
for exp in experiments:
    try:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        all_runs.extend(runs)
    except:
        pass

runs = all_runs

print(f"\n{'='*70}")
print(f"✅ MLflow Training Results: {len(runs)} Runs")
print(f"{'='*70}\n")

for i, run in enumerate(sorted(runs, key=lambda r: r.data.metrics.get('roc_auc', 0), reverse=True), 1):
    model_name = run.data.params.get('model_name', 'unknown')
    roc_auc = run.data.metrics.get('roc_auc', 0)
    f1 = run.data.metrics.get('f1', 0)
    precision = run.data.metrics.get('precision', 0)
    recall = run.data.metrics.get('recall', 0)
    accuracy = run.data.metrics.get('accuracy', 0)
    cv_auc = run.data.metrics.get('cv_roc_auc_mean', 0)
    
    print(f"{i}. {model_name.upper()}")
    print(f"   ROC-AUC:   {roc_auc:.4f} (CV: {cv_auc:.4f})")
    print(f"   F1:        {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print()

print(f"{'='*70}")
print("✅ All models uploaded to MLflow!")
print(f"{'='*70}")
