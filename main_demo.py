import numpy as np
import xgboost as xgb
import datetime
import hashlib
import json

# ==============================
# Load Model + Data
# ==============================
print("🔹 Loading model and data...")

X_test = np.load("X_test_enc.npy")
y_test = np.load("y_test.npy")

model = xgb.XGBClassifier()
model.load_model("ddos_xgb_model.json")

# ==============================
# Blockchain Simulation
# ==============================
blockchain = []

def store_on_blockchain(log):
    record = str(log)
    hash_val = hashlib.sha256(record.encode()).hexdigest()

    block = {
        "data": log,
        "hash": hash_val
    }

    blockchain.append(block)

    print("🔗 Stored on Algorand (simulated)")
    print("Hash:", hash_val[:12], "...")

# ==============================
# Logging
# ==============================
def log_attack(log):
    with open("logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")

# ==============================
# DPDP Compliance Check
# ==============================
def check_dpdp(log):
    if log["prediction"] == "DDoS" and log["data_at_risk"]:
        print("DPDP ALERT: Data breach detected")

# ==============================
# Real-time Simulation
# ==============================
print("\nStarting Real-Time Detection...\n")

for i in range(5):  # simulate 5 live samples
    sample = X_test[i].reshape(1, -1)

    print("Analyzing network traffic...")

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    log = {
        "timestamp": str(datetime.datetime.now()),
        "id": int(i),
        "prediction": "DDoS" if pred == 1 else "Benign",
        "confidence": float(prob),
        "data_at_risk": True if pred == 1 else False
    }

    # ==========================
    # OUTPUT (IMPORTANT)
    # ==========================
    if pred == 1:
        print(" DDoS ATTACK DETECTED")
    else:
        print("NORMAL TRAFFIC")

    print(f"Confidence: {prob:.2f}")

    # ==========================
    # SYSTEM ACTIONS
    # ==========================
    log_attack(log)
    check_dpdp(log)
    store_on_blockchain(log)

    print("-" * 50)