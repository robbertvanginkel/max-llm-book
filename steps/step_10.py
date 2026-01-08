"""
Step 10: Stacking Transformer Blocks

Stack multiple transformer blocks with embeddings to create
the complete GPT-2 model architecture.

Tasks:
1. Import Tensor, Embedding, Module, Sequential, and previous components
2. Create token and position embeddings
3. Stack n_layer transformer blocks using Sequential
4. Create final layer normalization
5. Implement forward pass: embeddings -> blocks -> layer norm

Run: pixi run s10
"""

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding, Module, Sequential
from .step_01 import GPT2Config
from .step_08 import LayerNorm
from .step_09 import GPT2Block 

class GPT2Model(Module):
    """Complete GPT-2 transformer model."""

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids):
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Hidden states, shape [batch, seq_length, n_embd]
        """
        batch_size, seq_length = input_ids.shape
        tok_embeds = self.wte(input_ids)
        position_indices = Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        pos_embeds = self.wpe(position_indices)
        x = tok_embeds + pos_embeds
        x = self.h(x)
        return self.ln_f(x)
