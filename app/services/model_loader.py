import torch
import json
from app.models.gru import GRUAnomalyDetector
from app.models.lstm_rnn import LSTM_RNN_Hybrid
from app.models.isolation_forest import IsolationForestWrapper


def load_gru_with_guess(path, input_size):
    best_first = {"hidden_size": 64, "num_layers": 1}
    tried_configs = [best_first] + [
        {"hidden_size": 128, "num_layers": 1},
        {"hidden_size": 64, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 2},
    ]

    for config in tried_configs:
        print(f"🔍 Trying GRU config: {config}")
        try:
            model = GRUAnomalyDetector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"✅ GRU model loaded successfully with config: {config}")
            with open("models/gru_metadata.json", "w") as f:
                json.dump(config, f)
            return model
        except Exception as e:
            print(f"❌ Failed with config {config}: {e}")
    raise ValueError("❌ Could not load GRU model with any known config.")

def load_lstm_with_guess(path, input_size):
    best_first = {"hidden_size": 128, "num_layers": 2}
    tried_configs = [best_first] + [
        {"hidden_size": 64, "num_layers": 1},
        {"hidden_size": 64, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 1},
    ]

    for config in tried_configs:
        print(f"🔍 Trying LSTM+RNN config: {config}")
        try:
            model = LSTM_RNN_Hybrid(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"✅ LSTM+RNN model loaded successfully with config: {config}")
            with open("models/lstm_metadata.json", "w") as f:
                json.dump(config, f)
            return model
        except Exception as e:
            print(f"❌ Failed with config {config}: {e}")
    raise ValueError("❌ Could not load LSTM+RNN model with any known config.")



def load_isolation_forest(model_path: str):
    return IsolationForestWrapper(model_path)

