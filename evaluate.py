import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb


MODEL_PATH = Path("ddos_xgb_model.json")
TEST_DATA_PATH = Path("X_test_enc.npy")
LOG_PATH = Path("logs.json")
PREDICTION_DELAY_SECONDS = 0.1


def load_model(model_path: Path = MODEL_PATH) -> xgb.XGBClassifier:
    """Load the trained XGBoost model from disk."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def predict_sample(model: xgb.XGBClassifier, sample: np.ndarray) -> tuple[str, float]:
    """Predict a single sample and return the label with confidence."""
    sample_2d = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    probabilities = model.predict_proba(sample_2d)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    prediction = "DDoS Attack" if predicted_class == 1 else "Normal"
    return prediction, confidence


def log_event(
    log_path: Path,
    timestamp: str,
    prediction: str,
    confidence: float,
    data_at_risk: bool,
) -> dict:
    """Append the current event to logs.json and return the saved entry."""
    entry = {
        "timestamp": timestamp,
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "data_at_risk": data_at_risk,
    }

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(entry)

    with log_path.open("w", encoding="utf-8") as file:
        json.dump(logs, file, indent=2)

    return entry


def check_compliance(prediction: str) -> bool:
    """Apply DPDP compliance logic to the prediction result."""
    data_at_risk = prediction == "DDoS Attack"
    if data_at_risk:
        print("DPDP ALERT: Data breach detected")
    return data_at_risk


def store_blockchain(blockchain: list[str], log_entry: dict) -> str:
    """Simulate storing the log entry on a blockchain by hashing it."""
    payload = json.dumps(log_entry, sort_keys=True).encode("utf-8")
    block_hash = hashlib.sha256(payload).hexdigest()
    blockchain.append(block_hash)
    print(f"Stored on Algorand blockchain (simulated): {block_hash}")
    return block_hash


def run_realtime_detection() -> None:
    """Simulate a real-time cybersecurity monitoring system."""
    print("Initializing real-time DDoS detection system...")
    print(f"Loading XGBoost model from {MODEL_PATH}...")
    model = load_model()

    print(f"Loading encoded test data from {TEST_DATA_PATH}...")
    test_samples = np.load(TEST_DATA_PATH)

    blockchain: list[str] = []

    with LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump([], file, indent=2)

    print(f"Prediction logging enabled: {LOG_PATH}")
    print("Real-time monitoring started.\n")

    for index, sample in enumerate(test_samples, start=1):
        timestamp = datetime.now().isoformat(timespec="seconds")
        prediction, confidence = predict_sample(model, sample)

        print(f"[{timestamp}] Sample {index}/{len(test_samples)}")
        print(f"Traffic Status : {prediction}")
        print(f"Confidence     : {confidence:.4f}")

        data_at_risk = check_compliance(prediction)
        log_entry = log_event(LOG_PATH, timestamp, prediction, confidence, data_at_risk)
        store_blockchain(blockchain, log_entry)
        print("-" * 70)

        time.sleep(PREDICTION_DELAY_SECONDS)

    print("\nMonitoring complete.")
    print(f"Total samples processed : {len(test_samples)}")
    print(f"Total blockchain records: {len(blockchain)}")
    print(f"Logs saved to           : {LOG_PATH.resolve()}")


if __name__ == "__main__":
    run_realtime_detection()
