import os
import sys
import mlflow


THRESHOLD = 0.85


def main():
    info_file = "model_info.txt"
    if not os.path.exists(info_file):
        print(f"ERROR: {info_file} not found.")
        sys.exit(1)

    with open(info_file, "r") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    print(f"Run ID: {run_id}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        import pathlib
        mlruns_dir = pathlib.Path(__file__).parent / "mlruns"
        mlflow.set_tracking_uri(mlruns_dir.as_uri())

    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"ERROR: Could not fetch run {run_id}: {e}")
        sys.exit(1)

    metrics = run.data.metrics
    accuracy = metrics.get("accuracy") or metrics.get("best_val_accuracy")

    if accuracy is None:
        print("ERROR: No 'accuracy' or 'best_val_accuracy' metric found for this run.")
        sys.exit(1)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Threshold:      {THRESHOLD}")

    if accuracy < THRESHOLD:
        print(f"\n✗ FAILED: Accuracy {accuracy:.4f} is below the threshold {THRESHOLD}.")
        print("  Pipeline will NOT proceed to deployment.")
        sys.exit(1)
    else:
        print(f"\n✓ PASSED: Accuracy {accuracy:.4f} meets the threshold {THRESHOLD}.")
        print("  Proceeding to deployment.")
        sys.exit(0)


if __name__ == "__main__":
    main()
