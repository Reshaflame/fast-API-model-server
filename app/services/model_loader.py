import torch
from app.models.gru import GRUAnomalyDetector
from app.models.lstm_rnn import LSTM_RNN_Hybrid

def load_gru(model_path: str, input_size: int, hidden_size=64, num_layers=2):
    model = GRUAnomalyDetector(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def load_lstm(model_path: str, input_size: int, hidden_size=64, num_layers=2):
    model = LSTM_RNN_Hybrid(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Mock Isolation Forest (returns 0 always)
def load_isolation_forest():
    class MockIF:
        def predict(self, X):
            return [0] * len(X)  # 0 = normal (not anomalous)
        def decision_function(self, X):
            return [0.0] * len(X)  # Always neutral
    return MockIF()
