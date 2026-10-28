"""Reference architecture extracted from training notebook cell 17.

Encoder keeps original top-level parameter names, so classifier state_dict keys
remain compatible with the original notebook. No substitute Mamba implementation.
"""
import torch
from torch import nn
try:
    from mamba_ssm import Mamba
except (ImportError, OSError) as exc:
    raise ImportError("Mamba requires compatible mamba-ssm and causal-conv1d CUDA builds. "
                      "See README; data/evaluation/CPU adapter tests do not require Mamba.") from exc

class BidirectionalMambaBlock(nn.Module):
    """A pre-norm, residual Mamba block that scans both directions and concatenates.

    Parameters
    ----------
    d_model : int
        Residual-stream / embedding dimension.
    d_state : int
        SSM state expansion factor (official `mamba_ssm.Mamba` parameter).
    d_conv : int
        Local causal-convolution kernel width.
    expand : int
        Inner block expansion factor.
    dropout : float
        Dropout probability applied to the block's output before the residual add.
    bidirectional : bool
        If True, run a second Mamba over the time-reversed sequence and concatenate
        + project back to `d_model`. If False, behaves like the v1 unidirectional block.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.merge = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm -> (bi-)Mamba -> dropout -> residual add.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch, length, d_model)`.

        Returns
        -------
        torch.Tensor
            Same shape as `x`.
        """
        residual = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        if self.bidirectional:
            out_bwd = self.mamba_bwd(torch.flip(x, dims=[1]))
            out_bwd = torch.flip(out_bwd, dims=[1])
            out = self.merge(torch.cat([out_fwd, out_bwd], dim=-1))
        else:
            out = out_fwd
        out = self.dropout(out)
        return residual + out

class AttentionPool(nn.Module):
    """Learned additive attention pooling over the time dimension.

    Parameters
    ----------
    d_model : int
        Feature dimension of the sequence to pool.
    hidden_dim : int
        Hidden width of the attention scorer MLP.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool `(batch, length, d_model)` -> `(batch, d_model)` via a softmax-weighted sum."""
        scores = self.scorer(x)                       # (B, T, 1)
        weights = torch.softmax(scores, dim=1)         # (B, T, 1), sums to 1 over time
        return (x * weights).sum(dim=1)

class ClassificationHead(nn.Module):
    """Two-layer MLP classification head producing raw (pre-sigmoid) logits.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (pooled backbone width).
    hidden_dim : int
        Hidden-layer width.
    n_classes : int
        Number of output logits (one per diagnostic superclass).
    dropout : float
        Dropout probability applied after the hidden activation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map pooled features to class logits. `x`: `(batch, in_dim)` -> `(batch, n_classes)`."""
        return self.net(x)

class MambaEncoder(nn.Module):
    """Head-free reference encoder: input projection -> bi-Mamba stack -> pooled features.

    The original constructor signature is retained for straightforward extraction;
    n_classes and hidden_dim are unused here and used by MambaClassifier's head.

    Parameters
    ----------
    n_leads : int
        Number of input channels (ECG leads).
    n_classes : int
        Number of output classes (multi-label).
    embed_dim : int
        Mamba residual-stream width (`d_model`).
    hidden_dim : int
        Classification-head hidden width.
    n_layers : int
        Number of stacked `BidirectionalMambaBlock`s.
    d_state, d_conv, expand : int
        Forwarded to every `mamba_ssm.Mamba` instance.
    dropout : float
        Dropout probability used in both the backbone blocks and the head.
    bidirectional : bool
        Whether Mamba blocks scan both directions (v2 default: True).
    pooling : str
        One of `{"mean", "mean_attention"}`. `"mean_attention"` (v2 default) concatenates
        plain mean-pooling with learned attention-pooling before the head.
    """

    def __init__(self, n_leads: int, n_classes: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, d_state: int, d_conv: int, expand: int, dropout: float,
                 bidirectional: bool = True, pooling: str = "mean_attention"):
        super().__init__()
        if pooling not in {"mean", "mean_attention"}:
            raise ValueError(f"pooling must be 'mean' or 'mean_attention', got {pooling!r}")
        self.pooling = pooling

        self.input_proj = nn.Linear(n_leads, embed_dim)
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(embed_dim, d_state, d_conv, expand, dropout, bidirectional)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if pooling == "mean_attention":
            self.attn_pool = AttentionPool(embed_dim)
            head_in_dim = embed_dim * 2   # mean ‖ attention-pooled
        else:
            self.attn_pool = None
            head_in_dim = embed_dim

        self.feature_dim = head_in_dim

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Existing backbone before pooling, shape (batch, time, embed_dim)."""
        x = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of ECG signals, including the reference pooling dropout.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch, n_leads, length)` — the raw preprocessed signal layout
            produced by Notebook 02's `PTBXLClassificationDataset`.

        Returns
        -------
        torch.Tensor
            Features of shape `(batch, feature_dim)`, with no classification head.
        """
        x = self.forward_tokens(x)

        mean_pooled = x.mean(dim=1)    # (B, embed_dim)
        if self.pooling == "mean_attention":
            attn_pooled = self.attn_pool(x)              # (B, embed_dim)
            pooled = torch.cat([mean_pooled, attn_pooled], dim=-1)  # (B, 2*embed_dim)
        else:
            pooled = mean_pooled

        pooled = self.dropout(pooled)
        return pooled

class MambaClassifier(MambaEncoder):
    """Original logits path with a reusable head-free encoder/forward_features."""
    def __init__(self, n_leads, n_classes, embed_dim, hidden_dim, n_layers,
                 d_state, d_conv, expand, dropout, bidirectional=True,
                 pooling="mean_attention"):
        super().__init__(n_leads, n_classes, embed_dim, hidden_dim, n_layers,
                         d_state, d_conv, expand, dropout, bidirectional, pooling)
        self.head = ClassificationHead(self.feature_dim, hidden_dim, n_classes, dropout)

    def forward_features(self, x):
        return super().forward(x)

    def forward(self, x):
        return self.head(self.forward_features(x))
