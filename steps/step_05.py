"""
Step 05: Token Embeddings

Implement token embeddings that convert discrete token IDs into continuous vectors.

Tasks:
1. Import Embedding and Module from max.nn.module_v3
2. Create token embedding layer using Embedding(vocab_size, dim=n_embd)
3. Implement forward pass that looks up embeddings for input token IDs

Run: pixi run s05
"""

from max.nn.module_v3 import Embedding, Module

from solutions.solution_01 import GPT2Config


class GPT2Embeddings(Module):
    """Token embeddings for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.wte = Embedding(config.vocab_size, dim=config.n_embd)

    def __call__(self, input_ids):
        """Convert token IDs to embeddings.

        Args:
            input_ids: Tensor of token IDs, shape [batch_size, seq_length]

        Returns:
            Token embeddings, shape [batch_size, seq_length, n_embd]
        """
        return self.wte(input_ids)
