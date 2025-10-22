import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.lightgbm.autolog()
mlflow.set_experiment("Penguins Classification")

train = pd.read_csv("data/train3.csv")
test = pd.read_csv("data/test3.csv")
X_train, y_train = train.drop("species", axis=1), train["species"]
X_test, y_test = test.drop("species", axis=1), test["species"]

os.makedirs("models_lgbm", exist_ok=True)

with mlflow.start_run(run_name="LightGBM Autolog"):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, average='macro'),
        "recall": recall_score(y_test, y_test_pred, average='macro'),
        "f1_score": f1_score(y_test, y_test_pred, average='macro')
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    model.booster_.save_model("models_lgbm/lgbm_model.txt")

print(" LightGBM results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
