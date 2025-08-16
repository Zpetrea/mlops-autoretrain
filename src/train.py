import os
import json
import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow

# Point MLflow to DagsHub unless overridden by env
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/Zpetrea/mlops-autoretrain.mlflow",
)
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("autoretrain")


def latest_csv(pattern: str = "data/raw/classification_*.csv") -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No raw data found under data/raw/")
    # Filenames have YYYYMMDD — lexicographic max is the latest
    return max(files)


def load_xy(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def main():
    csv_path = latest_csv()
    X, y = load_xy(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    n_estimators = int(os.getenv("N_ESTIMATORS", "200"))
    rnd_state = int(os.getenv("RANDOM_STATE", "123"))

    with mlflow.start_run():
        # Params
        mlflow.log_param("data_csv", os.path.basename(csv_path))
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", rnd_state)

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=rnd_state, n_jobs=-1
        )
        model.fit(X_tr, y_tr)

        # Metrics
        preds = model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds)

        # Save metrics for promotion
        Path("reports").mkdir(exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump({"accuracy": float(acc), "f1": float(f1)}, f)

        # Save candidate model artifacts
        out_dir = Path("artifacts/current_model")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / "model.pkl")
        with open(out_dir / "readme.txt", "w") as f:
            f.write(
                f"RandomForest trained on {os.path.basename(csv_path)}\n"
                f"acc={acc:.4f} f1={f1:.4f}\n"
            )

        # Log to MLflow (artifacts only — avoid the unsupported endpoint)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifacts(str(out_dir), artifact_path="candidate_model")

        print(
            f"Trained on {os.path.basename(csv_path)} | acc={acc:.4f} f1={f1:.4f}"
        )


if __name__ == "__main__":
    main()
