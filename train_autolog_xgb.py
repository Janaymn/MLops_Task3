import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.xgboost.autolog()
mlflow.set_experiment("Penguins Classification")

train = pd.read_csv("data/train3.csv")
test = pd.read_csv("data/test3.csv")
X_train, y_train = train.drop("species", axis=1), train["species"]
X_test, y_test = test.drop("species", axis=1), test["species"]

os.makedirs("models_xgb", exist_ok=True)

with mlflow.start_run(run_name="XGBoost Autolog"):
    model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

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

    model.save_model("models_xgb/xgb_model.json")

print(" XGBoost results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
