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
    def __init__(self, base_weight=None, base_bias=None, rank=1, alpha=1.0):
        super().__init__()
        self.base_weight = nn.Parameter(base_weight.clone(), requires_grad=False)
        self.base_bias   = nn.Parameter(base_bias.clone(),   requires_grad=False)

        # LoRA parameters (initialized near-zero)
        self.A = nn.Parameter(torch.zeros(rank, 3))
        self.B = nn.Parameter(torch.zeros(3, rank))
        self.scale = alpha / rank     # like LoRA α

    def forward(self, x):
        delta_w = self.scale * (self.B @ self.A)      # (3,3) low-rank update
        logits  = x @ (self.base_weight + delta_w).T + self.base_bias
        return torch.sigmoid(logits)