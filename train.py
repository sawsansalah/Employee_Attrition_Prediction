import argparse
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser("Employee Attrition – MLflow demo")
    p.add_argument("--csv", default="data/employee_data.csv", help="Path to CSV")
    p.add_argument("--target", default="left_company", help="Target column name")
    p.add_argument("--experiment", default="employee-attrition", help="MLflow experiment name")
    p.add_argument("--run", default="logreg-run", help="MLflow run name")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()

    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:7006")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    # Load data
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    with mlflow.start_run(run_name=args.run):

        # Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluation
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # Log params
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Save & log model
        os.makedirs("models", exist_ok=True)
        model_path = "models/attrition_model.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(f"✅ Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
