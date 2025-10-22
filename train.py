# train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from dotenv import load_dotenv
import os

# Load environment variables (MLflow tracking info)
load_dotenv()

mlflow.set_experiment("Penguins Classification")

# Load training data
train = pd.read_csv("data/train3.csv")
X = train.drop("species", axis=1)
y = train["species"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/log_reg_penguins.joblib")

# Log metrics and model to MLflow
with mlflow.start_run(run_name="logistic_regression_penguins"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_artifact("models/log_reg_penguins.joblib")

print(f"Model trained with accuracy: {acc:.4f}")
