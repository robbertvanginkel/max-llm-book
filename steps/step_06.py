"""
Step 06: Position Embeddings

Implement position embeddings that encode sequence order information.

Tasks:
1. Import Tensor from max.experimental.tensor
2. Import Embedding and Module from max.nn.module_v3
3. Create position embedding layer using Embedding(n_positions, dim=n_embd)
4. Implement forward pass that looks up embeddings for position indices

Run: pixi run s06
"""

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding, Module

from solutions.solution_01 import GPT2Config


class GPT2PositionEmbeddings(Module):
    """Position embeddings for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.wpe = Embedding(config.n_positions, dim=config.n_embd)

    def __call__(self, position_ids):
        """Convert position indices to embeddings.

        Args:
            position_ids: Tensor of position indices, shape [seq_length] or [batch_size, seq_length]

        Returns:
            Position embeddings, shape matching input with added embedding dimension
        """
        return self.wpe(position_ids)
