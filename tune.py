# tune.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

mlflow.set_experiment("Penguins Classification")

# Load train/test data
train = pd.read_csv("data/train3.csv")
test = pd.read_csv("data/test3.csv")
X_train, y_train = train.drop("species", axis=1), train["species"]
X_test, y_test = test.drop("species", axis=1), test["species"]

# Parameter grid
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

best_acc = 0
best_params = {}

with mlflow.start_run(run_name="rf_tuning_penguins"):
    for params in ParameterGrid(param_grid):
        with mlflow.start_run(nested=True, run_name=f"rf_{params['n_estimators']}_{params['max_depth']}"):
            rf = RandomForestClassifier(**params, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_params(params)
            mlflow.log_metric("test_accuracy", acc)

            os.makedirs("models", exist_ok=True)
            path = f"models/rf_{params['n_estimators']}_{params['max_depth']}.joblib"
            joblib.dump(rf, path)
            mlflow.log_artifact(path)

            if acc > best_acc:
                best_acc = acc
                best_params = params

    mlflow.log_metric("best_accuracy", best_acc)
    mlflow.log_param("best_params", str(best_params))

print(f"Best model: {best_params} with accuracy {best_acc:.3f}")
