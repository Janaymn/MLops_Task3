# train_autolog_lgbm.py

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
import mlflow
from dotenv import load_dotenv

# Load .env (for DagsHub MLflow remote)
load_dotenv()

# Enable autologging for LightGBM
mlflow.lightgbm.autolog()

# Set experiment name
mlflow.set_experiment("Penguins Classification")

# Load pre-split data
train = pd.read_csv("data/train3.csv")
test = pd.read_csv("data/test3.csv")

X_train = train.drop("species", axis=1)
y_train = train["species"]
X_test = test.drop("species", axis=1)
y_test = test["species"]

# Start MLflow run
with mlflow.start_run(run_name="LightGBM Autolog"):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Extra manual metrics
    y_pred = model.predict(X_test)
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='macro'))

print("LightGBM autolog run logged successfully on Penguins dataset.")
