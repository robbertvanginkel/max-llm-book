"""
Step 07: Multi-head Attention

Implement multi-head attention that splits Q/K/V into multiple heads,
computes attention in parallel for each head, and merges the results.

Tasks:
1. Import required modules (math, F, Tensor, Linear, Module, etc.)
2. Create c_attn and c_proj linear layers
3. Implement _split_heads: reshape and transpose to add head dimension
4. Implement _merge_heads: transpose and reshape to remove head dimension
5. Implement _attn: compute attention for all heads in parallel
6. Implement forward pass: project -> split -> attend -> merge -> project

Run: pixi run s07
"""

# TODO: Import required modules
# Hint: You'll need math for scaling
# Hint: You'll need functional as F from max.experimental
# Hint: You'll need Tensor, Device, DType from max.experimental.tensor and max.driver
# Hint: You'll need Dim, DimLike from max.graph
# Hint: You'll also need Linear and Module from max.nn.module_v3

import math
from max.experimental import functional as F
from max.nn.module_v3 import Linear, Module
from numpy import dtype
from torch import device

from .step_01 import GPT2Config


# TODO: Copy causal_mask function from solution_02.py
# This is the same function you implemented in Step 02
from steps.step_02 import causal_mask

class GPT2MultiHeadAttention(Module):
    """Multi-head attention for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)

        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size).

        Args:
            tensor: Input tensor, shape [batch, seq_length, n_embd]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, num_heads, seq_length, head_size]
        """
        tensor = tensor.reshape(
            tensor.shape[:-1] + [num_heads, attn_head_size]
        )

        return tensor.transpose(-3, -2)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back to original shape.

        Args:
            tensor: Input tensor, shape [batch, num_heads, seq_length, head_size]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, seq_length, n_embd]
        """
        tensor = tensor.transpose(-3, -2)
        return tensor.reshape(tensor.shape[:-2] + [num_heads * attn_head_size])

    def _attn(self, query, key, value):
        """Compute attention for all heads in parallel.

        Args:
            query: Query tensor, shape [batch, num_heads, seq_length, head_size]
            key: Key tensor, shape [batch, num_heads, seq_length, head_size]
            value: Value tensor, shape [batch, num_heads, seq_length, head_size]

        Returns:
            Attention output, shape [batch, num_heads, seq_length, head_size]
        """
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))
        attn_weights = attn_weights + causal_mask(query.shape[-2], 0, dtype=query.dtype, device=query.device)
        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value
        return attn_output

    def __call__(self, hidden_states):
        """Apply multi-head attention.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Attention output, shape [batch, seq_length, n_embd]
        """
        # TODO: Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        return self.c_proj(attn_output)
