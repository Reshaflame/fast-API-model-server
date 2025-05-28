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
        self.model[0].weight.data = torch.tensor([[0.5, 0.5, 0.0]])
        self.model[0].bias.data.fill_(0.0)

    def forward(self, x):
        return self.model(x)
