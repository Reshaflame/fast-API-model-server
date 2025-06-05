import torch
import torch.nn as nn

class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(GRUAnomalyDetector, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.contiguous()
        h, _ = self.gru(x)
        out = self.fc(h[:, -1, :])
        return self.sigmoid(out)
