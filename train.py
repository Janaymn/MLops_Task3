# train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.set_experiment("Penguins Classification")

# Load training data
train = pd.read_csv("data/train3.csv")
test = pd.read_csv("data/test3.csv")

X_train, y_train = train.drop("species", axis=1), train["species"]
X_test, y_test = test.drop("species", axis=1), test["species"]

os.makedirs("models", exist_ok=True)

with mlflow.start_run(run_name="logistic_regression_penguin"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, average='macro'),
        "recall": recall_score(y_test, y_test_pred, average='macro'),
        "f1_score": f1_score(y_test, y_test_pred, average='macro')
    }

    # Log metrics
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    joblib.dump(model, "models/log_reg_penguins.joblib")
    mlflow.log_artifact("models/log_reg_penguins.joblib")

print(" Logistic Regression results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
