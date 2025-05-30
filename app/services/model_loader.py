import torch
import json
from app.models.gru import GRUAnomalyDetector
from app.models.lstm_rnn import LSTM_RNN_Hybrid
from app.models.isolation_forest import IsolationForestWrapper


def load_gru_with_guess(model_path: str, input_size: int):
    # Try the original search grid you used during training
    search_grid = [
        {"hidden_size": 64, "num_layers": 1},
        {"hidden_size": 128, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 1}
    ]

    for config in search_grid:
        try:
            print(f"🔍 Trying GRU config: {config}")
            model = GRUAnomalyDetector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            print(f"✅ GRU model loaded successfully with config: {config}")

            # ✅ Save the config for future use
            with open("models/gru_metadata.json", "w") as f:
                json.dump({
                    "input_size": input_size,
                    **config
                }, f)

            return model

        except RuntimeError as e:
            print(f"❌ Failed with config {config}: {e}")
            continue

    raise RuntimeError("❌ Could not load GRU model with any known config.")

def load_lstm_with_guess(model_path: str, input_size: int):
    # Same search grid used in your training pipeline
    search_grid = [
        {"hidden_size": 64, "num_layers": 1},
        {"hidden_size": 128, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 1}
    ]

    for config in search_grid:
        try:
            print(f"🔍 Trying LSTM+RNN config: {config}")
            model = LSTM_RNN_Hybrid(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            print(f"✅ LSTM+RNN model loaded successfully with config: {config}")

            # ✅ Save the config for future runs
            with open("models/lstm_metadata.json", "w") as f:
                json.dump({
                    "input_size": input_size,
                    **config
                }, f)

            return model

        except RuntimeError as e:
            print(f"❌ Failed with config {config}: {e}")
            continue

    raise RuntimeError("❌ Could not load LSTM+RNN model with any known config.")


def load_isolation_forest(model_path: str):
    return IsolationForestWrapper(model_path)

