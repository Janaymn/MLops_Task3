# tune.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import joblib
import os
from dotenv import load_dotenv

# Load environment variables for DagsHub MLflow
load_dotenv()

mlflow.set_experiment("Penguins Classification")

# Load train/test data
train = pd.read_csv("data/train3.csv")
test = pd.read_csv("data/test3.csv")

X_train, y_train = train.drop("species", axis=1), train["species"]
X_test, y_test = test.drop("species", axis=1), test["species"]

# Parameter grid for tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

best_acc = 0
best_params = {}
best_metrics = {}

# Outer run to group all tuning experiments
with mlflow.start_run(run_name="rf_tuning_penguins"):
    for params in ParameterGrid(param_grid):
        # Nested run for each parameter combination
        with mlflow.start_run(nested=True, run_name=f"rf_{params['n_estimators']}_{params['max_depth']}"):
            rf = RandomForestClassifier(**params, random_state=42)
            rf.fit(X_train, y_train)

            # Predictions
            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)

            # Compute metrics
            metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, average='macro'),
                "recall": recall_score(y_test, y_test_pred, average='macro'),
                "f1_score": f1_score(y_test, y_test_pred, average='macro')
            }

            # Log parameters and metrics to MLflow
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Save model artifact
            os.makedirs("models_tuned", exist_ok=True)
            path = f"models_tuned/rf_{params['n_estimators']}_{params['max_depth']}.joblib"
            joblib.dump(rf, path)
            mlflow.log_artifact(path)

            # Track best-performing model
            if metrics["test_accuracy"] > best_acc:
                best_acc = metrics["test_accuracy"]
                best_params = params
                best_metrics = metrics

            # Print per-model results
            print(f"\n Model rf_{params['n_estimators']}_{params['max_depth']}")
            for k, v in metrics.items():
                print(f"   {k}: {v:.4f}")

    # Log best model results in parent run
    mlflow.log_metric("best_accuracy", best_acc)
    mlflow.log_param("best_params", str(best_params))

print("\n Best Random Forest Model:")
print(f"Parameters: {best_params}")
for k, v in best_metrics.items():
    print(f"{k}: {v:.4f}")
