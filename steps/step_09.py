"""
Step 09: Transformer Block

Combine multi-head attention, MLP, layer normalization, and residual
connections into a complete transformer block.

Tasks:
1. Import Module and all previous solution components
2. Create ln_1, attn, ln_2, and mlp layers
3. Implement forward pass with pre-norm residual pattern

Run: pixi run s09
"""

from max.nn.module_v3 import Module
from .step_01 import GPT2Config
from .step_04 import GPT2MLP
from .step_07 import GPT2MultiHeadAttention
from .step_08 import LayerNorm

class GPT2Block(Module):
    """Complete GPT-2 transformer block."""

    def __init__(self, config: GPT2Config):
        """Initialize transformer block.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        hidden_size = config.n_embd
        inner_dim = (
            config.n_inner
            if hasattr(config, "n_inner") and config.n_inner is not None
            else 4 * hidden_size
        )

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2MultiHeadAttention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def __call__(self, hidden_states):
        """Apply transformer block.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Output tensor, shape [batch, seq_length, n_embd]
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states
