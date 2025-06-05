import torch
import torch.nn as nn

class EnsembleMLP(nn.Module):
    def __init__(self):
        super(EnsembleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 1),   # 3 inputs → 1 output
            nn.Sigmoid()
        )
        # 🧪 Initialize with fixed weights
        self.model[0].weight.data = torch.tensor([[1.0/3, 1.0/3, 1.0/3]])
        self.model[0].bias.data.fill_(0.0)

    def forward(self, x):
        return self.model(x)

class LoRAEnsemble(nn.Module):
    def __init__(self, base_weight, base_bias=None, rank=1, alpha=1.0):
        super().__init__()

        # baseline (frozen)
        self.register_buffer("base_weight", base_weight.view(1, 3))
        self.register_buffer("base_bias",   torch.zeros(1) if base_bias is None else base_bias.view(1))

        # low-rank ΔW  (trainable)
        self.A = nn.Parameter(torch.zeros(rank, 3))   # (r, 3)
        self.B = nn.Parameter(torch.zeros(1, rank))   # (1, r)
        self.delta_b = nn.Parameter(torch.zeros(1))
        self.scale = alpha / rank
        nn.init.normal_(self.A, std=0.02)

    def forward(self, x):  # x: [B, 3]
        delta_w = self.scale * (self.B @ self.A)  # (1, 3)
        w = self.base_weight + delta_w            # (1, 3)
        b = self.base_bias + self.delta_b         # scalar + learnable shift
        logits = x @ w.T + b                      # (B, 1)
        return torch.sigmoid(logits)

class DeepHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # GRU & ISO → 8 hidden
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)        # x shape [B, 2]