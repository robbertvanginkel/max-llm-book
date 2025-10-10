"""GPT-2 model configuration."""

from dataclasses import dataclass

@dataclass
class GPT2Config:
    """GPT-2 configuration matching HuggingFace.

    Attributes:
        vocab_size: Size of the vocabulary.
        n_positions: Maximum sequence length.
        n_embd: Embedding dimension.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_inner: Inner dimension of feed-forward network (defaults to 4 * n_embd if None).
        layer_norm_epsilon: Epsilon for layer normalization.
    """
    vocab_size: int = None
    n_positions: int = None
    n_embd: int = None
    n_layer: int = None
    n_head: int = None
    n_inner: int = None
    layer_norm_epsilon: float = None
