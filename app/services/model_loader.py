import torch
from app.models.gru import GRUAnomalyDetector

def load_model(model_path: str, input_size: int, hidden_size=64, num_layers=1):
    model = GRUAnomalyDetector(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model



# Mock loader:
# def load_model(model_name: str):
#     print(f"Mock loading model: {model_name}")
#     return None
