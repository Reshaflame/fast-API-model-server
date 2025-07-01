# app/models/gru_hybrid.py
import torch.nn as nn

class GRUAnomalyDetector(nn.Module):
    """
    Matches training-server version: GRU ➜ residual ‘booster’ ➜ FC.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        # 🟢 tiny booster – layer-norm + FC with residual
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fc       = nn.Linear(hidden_size, output_size)
        self.dropout  = nn.Dropout(0.2)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:                       # [B, F] → [B, 1, F]
            x = x.unsqueeze(1)
        h, _  = self.gru(x.contiguous())       # (B, T, H)
        h_last = h[:, -1]                      # last time-step
        h_last = h_last + self.post(h_last)    # residual boost
        out = self.fc(self.dropout(h_last))
        return self.sigmoid(out)               # same as training
