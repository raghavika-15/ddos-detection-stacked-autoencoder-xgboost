import pandas as pd

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "DrDoS_UDP.csv",
    "Syn.csv"
]

SELECTED_FEATURES = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    " Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Fwd Packet Length Mean",
    " Bwd Packet Length Mean",
    " Packet Length Mean",
    " Packet Length Std",
    " Active Mean",
    " Idle Mean"
]

CHUNK_SIZE = 100_000


def load_data():
    frames = []

    for file in FILES:
        print(f"Loading {file}")

        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, engine="python"):
            # Detect label column safely
            label_col = [c for c in chunk.columns if c.strip().lower() == "label"][0]

            # Convert labels
            chunk[label_col] = (
                chunk[label_col]
                .astype(str)
                .str.lower()
                .apply(lambda x: 0 if "benign" in x else 1)
            )

            # ✅ SAFE FEATURE SELECTION (KEY FIX)
            available_features = [f for f in SELECTED_FEATURES if f in chunk.columns]

            chunk = chunk[available_features + [label_col]]
            chunk.rename(columns={label_col: "Label"}, inplace=True)

            # Reduce memory
            for col in available_features:
                chunk[col] = chunk[col].astype("float32")

            frames.append(chunk)

            break  # sample only first chunk per file

    return pd.concat(frames, ignore_index=True)
