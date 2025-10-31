"""Train ML models and track carbon emissions."""
import time
import mlflow
import os
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score


def load_dataset(name):
    """Load dataset and return train/test split."""
    if name == 'iris':
        data = load_iris()
        task = 'classification'
    elif name == 'wine':
        data = load_wine()
        task = 'classification'
    elif name == 'housing':
        data = fetch_california_housing()
        task = 'regression'
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, task

def train_model(model, model_name, X_train, X_test, y_train, y_test, dataset_name, task):
    """Train model and track metrics + carbon."""
    print(f"  Training {model_name}...")
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Estimate CO2 (rough estimate: 0.0001 kg CO2 per second of compute)
    # This is a simplified model for demonstration
    co2_kg = training_time * 0.0001
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    if task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        metric_name = 'accuracy'
        metric_value = accuracy
    else:
        accuracy = r2_score(y_test, y_pred)
        f1 = None
        metric_name = 'r2_score'
        metric_value = accuracy
    
    # Estimate energy from CO2 (assuming 0.5 kg CO2/kWh)
    energy_kwh = co2_kg / 0.5
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"{dataset_name}_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_metric(metric_name, metric_value)
        if f1 is not None:
            mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.log_metric("co2_kg", co2_kg)
        mlflow.log_metric("energy_kwh", energy_kwh)
    
    print(f"    ✓ {metric_name}: {metric_value:.3f} | CO2: {co2_kg:.6f} kg | Time: {training_time:.2f}s")

def main():
    """Run all experiments."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("green_ai_comparison")
    
    datasets = {
        'iris': 'classification',
        'wine': 'classification',
        'housing': 'regression'
    }
    
    models_cls = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }
    
    models_reg = {
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    print("\nStarting ML Models\n")
    
    for dataset_name, task in datasets.items():
        print(f"{'='*50}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        X_train, X_test, y_train, y_test, task = load_dataset(dataset_name)
        models = models_cls if task == 'classification' else models_reg
        
        for model_name, model in models.items():
            train_model(model, model_name, X_train, X_test, y_train, y_test, dataset_name, task)
        print()
    
    print("All models complete!\n")


if __name__ == "__main__":
    main()