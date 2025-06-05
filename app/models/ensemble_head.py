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
        """
        base_weight : tensor shape (1, 3)
        base_bias   : tensor shape (1,)   (can be zeros)
        """
        super().__init__()

        # --- frozen baseline ---------------------------------
        self.base_weight = nn.Parameter(base_weight.clone(), requires_grad=False)
        if base_bias is None:
            base_bias = torch.zeros(1)
        self.base_bias   = nn.Parameter(base_bias.clone(),   requires_grad=False)

        # --- LoRA parameters (trainable) ---------------------
        #  (B @ A) must be (1, 3)  → choose  B: (1, r)   A: (r, 3)
        self.A = nn.Parameter(torch.zeros(rank, 3))
        self.B = nn.Parameter(torch.zeros(1, rank))
        self.scale = alpha / rank      # standard LoRA scaling

        # tiny init so it starts near-zero
        nn.init.normal_(self.A, std=0.01)
        nn.init.zeros_(self.B)

    def forward(self, x):              # x shape [B, 3]
        delta_w = self.scale * (self.B @ self.A)     # (1, 3)
        logits  = x @ (self.base_weight + delta_w).T + self.base_bias  # (B,1)
        return torch.sigmoid(logits)    # (B,1)