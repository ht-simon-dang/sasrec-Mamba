
import torch
from torch import nn
from mamba_ssm import Mamba

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss: -log σ(pos_score - neg_score)"""
    def __init__(self):
        super().__init__()

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        diff = pos_score - neg_score
        return -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, inner_size: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        # residual + norm
        return self.norm(h + x)


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        use_residual: bool = True
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model, inner_size=d_model * 4, dropout=dropout)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mamba(x)
        if self.use_residual:
            h = self.norm(self.dropout(h) + x)
        else:
            h = self.norm(self.dropout(h))
        return self.ffn(h)
