"""
FlashAttention Transformer Model for Trading

This module implements a Transformer model with optional FlashAttention support
for financial time-series prediction.

FlashAttention benefits:
- 2-4x faster training and inference
- 10-100x less memory usage
- Enables longer context windows (8K+ timesteps)
- Exact attention (no approximation)

Usage:
    >>> model = FlashAttentionTrader(input_dim=25, d_model=256, use_flash=True)
    >>> predictions = model(X)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# Try to import FlashAttention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    # FlashAttention not installed - will use standard attention


class FlashMultiHeadAttention(nn.Module):
    """
    Multi-head attention with FlashAttention support.

    Falls back to standard attention if:
    - FlashAttention is not installed
    - return_attention=True (FlashAttention doesn't return weights)
    - Running on CPU

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_flash: Whether to use FlashAttention when available
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash = use_flash and FLASH_AVAILABLE

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with FlashAttention or standard attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, 1, seq_len, seq_len]
            return_attention: Whether to return attention weights
                             (forces standard attention)

        Returns:
            Tuple of (output, attention_weights)
            attention_weights is None when using FlashAttention
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Check if we can use FlashAttention
        can_use_flash = (
            self.use_flash and
            not return_attention and
            x.is_cuda and
            mask is None  # FlashAttention has limited mask support
        )

        if can_use_flash:
            # Use FlashAttention
            # Expected shape: [batch, seq, n_heads, d_k]
            dropout_p = self.dropout.p if self.training else 0.0
            output = flash_attn_func(Q, K, V, dropout_p=dropout_p)
            output = output.view(batch_size, seq_len, self.d_model)
            attn_weights = None
        else:
            # Standard attention (fallback)
            Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len, self.d_model)

        output = self.out_proj(output)

        return output, attn_weights


class FlashTransformerBlock(nn.Module):
    """
    Transformer encoder block with FlashAttention.

    Uses pre-norm architecture (LayerNorm before attention/FFN)
    which is more stable for training.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        use_flash: Whether to use FlashAttention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.attention = FlashMultiHeadAttention(d_model, n_heads, dropout, use_flash)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attention(x, mask, return_attention)
        x = residual + attn_out

        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x, attn_weights


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for time series.

    Unlike fixed sinusoidal encoding, learnable positions can adapt
    to temporal patterns specific to financial data.
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        return self.dropout(x)


class FlashAttentionTrader(nn.Module):
    """
    Transformer model for trading with FlashAttention.

    This model uses FlashAttention to enable efficient processing
    of long time series, allowing for extended lookback periods
    that would be impractical with standard attention.

    Benefits for trading:
    - Handle 4K-16K historical timesteps
    - Faster training and inference
    - Lower memory usage
    - Same prediction quality (exact attention)

    Args:
        input_dim: Number of input features per timestep
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        max_seq_len: Maximum sequence length
        n_outputs: Number of output predictions
        output_type: 'regression', 'direction', or 'allocation'
        dropout: Dropout probability
        use_flash: Whether to use FlashAttention

    Example:
        >>> model = FlashAttentionTrader(
        ...     input_dim=25,
        ...     d_model=256,
        ...     n_heads=8,
        ...     n_layers=6,
        ...     n_outputs=5,
        ...     use_flash=True
        ... )
        >>> predictions, _ = model(X)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: Optional[int] = None,
        max_seq_len: int = 4096,
        n_outputs: int = 1,
        output_type: str = 'regression',
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_outputs = n_outputs
        self.output_type = output_type
        self.use_flash = use_flash and FLASH_AVAILABLE

        if d_ff is None:
            d_ff = 4 * d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers with FlashAttention
        self.layers = nn.ModuleList([
            FlashTransformerBlock(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

        # Output head based on task type
        if output_type == 'regression':
            self.head = nn.Linear(d_model, n_outputs)
        elif output_type == 'direction':
            # Binary classification (up/down)
            self.head = nn.Linear(d_model, n_outputs)
        elif output_type == 'allocation':
            # Portfolio weights bounded by tanh
            self.head = nn.Sequential(
                nn.Linear(d_model, n_outputs),
                nn.Tanh()
            )
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights from all layers
                             (forces standard attention, useful for interpretability)

        Returns:
            Tuple of (predictions, attention_weights_list)
            - predictions: [batch, n_outputs]
            - attention_weights_list: List of attention weights per layer,
                                     or None if using FlashAttention
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer layers
        all_attention = [] if return_attention else None

        for layer in self.layers:
            x, attn = layer(x, mask, return_attention)
            if return_attention and attn is not None:
                all_attention.append(attn)

        x = self.norm(x)

        # Use last token for prediction (like CLS token)
        x = x[:, -1, :]

        # Output head
        predictions = self.head(x)

        return predictions, all_attention

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on output type.

        Args:
            predictions: Model predictions [batch, n_outputs]
            targets: Target values [batch, n_outputs]

        Returns:
            Loss tensor
        """
        if self.output_type == 'regression':
            return F.mse_loss(predictions, targets)
        elif self.output_type == 'direction':
            # Binary cross entropy for direction prediction
            return F.binary_cross_entropy_with_logits(
                predictions,
                (targets > 0).float()
            )
        elif self.output_type == 'allocation':
            # Maximize returns (negative returns as loss)
            # predictions are positions [-1, 1], targets are returns
            returns = predictions * targets
            return -torch.mean(returns)
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        This temporarily disables FlashAttention to obtain weights.
        Useful for understanding which historical periods the model
        focuses on for predictions.

        Args:
            x: Input tensor
            layer_idx: Which layer's attention to return (-1 for last)

        Returns:
            Attention weights [batch, n_heads, seq_len, seq_len]
        """
        # Temporarily disable FlashAttention
        original_use_flash = self.use_flash
        for layer in self.layers:
            layer.attention.use_flash = False

        try:
            _, all_attention = self.forward(x, return_attention=True)
            if all_attention:
                return all_attention[layer_idx]
            else:
                return None
        finally:
            # Restore FlashAttention setting
            for layer in self.layers:
                layer.attention.use_flash = original_use_flash


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive prediction.

    Args:
        seq_len: Sequence length
        device: Device for the tensor

    Returns:
        Causal mask [1, 1, seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)


if __name__ == '__main__':
    # Quick test
    print(f"FlashAttention available: {FLASH_AVAILABLE}")

    # Create model
    model = FlashAttentionTrader(
        input_dim=25,
        d_model=128,
        n_heads=8,
        n_layers=4,
        n_outputs=5,
        output_type='regression',
        use_flash=True
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using FlashAttention: {model.use_flash}")

    # Test forward pass
    batch_size = 4
    seq_len = 2048
    x = torch.randn(batch_size, seq_len, 25)

    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

    with torch.no_grad():
        output, _ = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed!")
