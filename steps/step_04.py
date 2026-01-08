"""
Step 04: Feed-forward Network (MLP)

Implement the MLP used in each transformer block with GELU activation.

Tasks:
1. Import functional (as F), Tensor, Linear, and Module from MAX
2. Create c_fc linear layer (embedding to intermediate dimension)
3. Create c_proj linear layer (intermediate back to embedding dimension)
4. Apply c_fc transformation in forward pass
5. Apply GELU activation function
6. Apply c_proj transformation and return result

Run: pixi run s04
"""
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from .step_01 import GPT2Config


class GPT2MLP(Module):
    """Feed-forward network matching HuggingFace GPT-2 structure.

    Args:
        intermediate_size: Size of the intermediate layer.
        config: GPT-2 configuration.
    """

    def __init__(self, intermediate_size: int, config: GPT2Config):
        super().__init__()
        embed_dim = config.n_embd

        # 2: Create the first linear layer (embedding to intermediate)
        self.c_fc = Linear(in_dim=embed_dim, out_dim=intermediate_size, bias=True)

        # 3: Create the second linear layer (intermediate back to embedding)
        self.c_proj = Linear(in_dim=intermediate_size, out_dim=embed_dim, bias=True)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply feed-forward network.

        Args:
            hidden_states: Input hidden states.

        Returns:
            MLP output.
        """
        # 4: Apply the first linear transformation
        hidden_states = self.c_fc(hidden_states)

        # 5: Apply GELU activation function
        hidden_states = F.gelu(hidden_states, approximate="tanh")

        # 6: Apply the second linear transformation and return
        return self.c_proj(hidden_states)
