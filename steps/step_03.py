"""
Step 03: Layer Normalization

Implement layer normalization that normalizes activations for training stability.

Tasks:
1. Import functional module (as F) and Tensor from max.experimental
2. Initialize learnable weight (gamma) and bias (beta) parameters
3. Apply layer normalization using F.layer_norm in the forward pass

Run: pixi run s03
"""

from max.experimental import functional as F
from max.experimental.tensor import Tensor

from max.graph import DimLike
from max.nn.module_v3 import Module


class LayerNorm(Module):
    """Layer normalization module.

    Args:
        dim: Dimension to normalize over.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # 2: Initialize learnable weight and bias parameters
        self.weight = Tensor.ones(shape=[dim])

        # TODO: Create self.bias as a Tensor of zeros with shape [dim]
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros
        # Hint: This is the beta parameter in layer normalization
        self.bias = Tensor.zeros(shape=[dim])

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        # 3: Apply layer normalization and return the result
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)
