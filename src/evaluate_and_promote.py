"""
Promote the newly trained candidate model if it beats the current best.

- Reads new metrics from reports/metrics.json (written by train.py)
- Compares to models/latest/metrics.json (if present)
- If improved, replaces models/latest/* with artifacts/current_model/*
  and writes models/latest/metrics.json with the new accuracy.
"""

import json
import shutil
from pathlib import Path

NEW_METRICS = Path("reports/metrics.json")
CANDIDATE_DIR = Path("artifacts/current_model")
BEST_DIR = Path("models/latest")
BEST_METRICS = BEST_DIR / "metrics.json"


def read_accuracy(path: Path, default: float = -1.0) -> float:
    try:
        with path.open() as f:
            data = json.load(f)
        return float(data.get("accuracy", default))
    except Exception:
        return default


def main() -> None:
    if not NEW_METRICS.exists():
        raise FileNotFoundError(
            "Missing reports/metrics.json. Run: python src/train.py"
        )
    if not CANDIDATE_DIR.exists():
        raise FileNotFoundError(
            "Missing artifacts/current_model/. Run: python src/train.py"
        )

    new_acc = read_accuracy(NEW_METRICS, default=-1.0)
    old_acc = read_accuracy(BEST_METRICS, default=-1.0)

    if new_acc > old_acc:
        # Replace best dir with the new candidate
        if BEST_DIR.exists():
            shutil.rmtree(BEST_DIR)
        shutil.copytree(CANDIDATE_DIR, BEST_DIR)

        # Write the updated best metrics
        BEST_DIR.mkdir(parents=True, exist_ok=True)
        with BEST_METRICS.open("w") as f:
            json.dump({"accuracy": new_acc}, f)

        print(f"PROMOTED: {old_acc} -> {new_acc}")
    else:
        print(f"NO PROMOTION: {old_acc} >= {new_acc}")


if __name__ == "__main__":
    main()
