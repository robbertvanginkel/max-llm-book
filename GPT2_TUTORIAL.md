# GPT-2 Implementation Tutorial with MAX

This tutorial guides you through building a complete GPT-2 model from scratch using Modular's MAX framework. Follow these steps sequentially to recreate the implementation in the `steps/` directory.

## Dependencies
- modular = "25.7.*"
- transformers = ">=4.57.0,<5"
- numpy = ">=2.3.3,<3"

## Overview

You'll build the following components in order:
1. Configuration dataclass
2. Utility functions (causal masking)
3. Layer Normalization
4. MLP
5. Multi-Head Attention
6. Transformer block
7. Complete GPT-2 model
8. Tokenizer wrapper
9. Text generation utilities
10. Demo script

---

## Step 1: Create Model Configuration (`config.py`)

**Purpose**: Define the GPT-2 model architecture parameters.

### Key Concepts:
**Dataclasses in Python**:
- Python's [`@dataclass`](https://docs.python.org/3/library/dataclasses.html) decorator reduces boilerplate code when creating configuration objects and provides clean syntax for defining class attributes with type hints and default values

**Model Configuration**:
- Configuration objects centralize hyperparameters and architecture settings in one place
- Makes it easy to experiment with different model sizes and settings
- Essential for reproducibility and model initialization

**Matching Hugging Face Models**:
- Using the same configuration values as [Hugging Face's pretrained GPT-2](https://huggingface.co/openai-community/gpt2) ensures weight compatibility
- Allows loading and using pretrained weights for inference without retraining
- Configuration values are accessed with the [transformers library](https://pypi.org/project/transformers/) in `hugging-face-model.py`

**GPT-2 Architecture Parameters**:
- vocab_size: Size of the token vocabulary (number of unique tokens the model can process)
- n_positions: Maximum sequence length (context window)
- n_embd: Embedding dimension (size of hidden states)
- n_layer: Number of transformer blocks stacked vertically
- n_head: Number of attention heads per layer
- n_inner: Dimension of the MLP intermediate layer (typically 4x n_embd)
- layer_norm_epsilon: Small constant for numerical stability in layer normalization

### Implementation Tasks (`config.py`):
1. Import dataclass from the dataclasses module 
2. Add the Python @dataclass decorator to the GPT2Config class
3. Run `pixi run huggingface` to get the correct values for the model parameters
4. Replace the None of the GPT2Config properties with the correct values

**Implementation**:
```python
from dataclasses import dataclass

@dataclass
class GPT2Config:
   # run `pixi run hugging-face-model` to get the correct values
    vocab_size: int = ?
    n_positions: int = ?
    n_embd: int = ?
    n_layer: int = ?
    n_head: int = ?
    n_inner: int = ?
    layer_norm_epsilon: float = ?
```

### Validation:
run `pixi run s01`

A failed test will show
```bash
✗ dataclass is not imported from dataclasses
✗ GPT2Config does not have the @dataclass decorator
✗ vocab_size is incorrect: expected match with Hugging Face model configuration, got None
✗ n_positions is incorrect: expected match with Hugging Face model configuration, got None
✗ n_embd is incorrect: expected match with Hugging Face model configuration, got None
✗ n_layer is incorrect: expected match with Hugging Face model configuration, got None
✗ n_head is incorrect: expected match with Hugging Face model configuration, got None
✗ n_inner is incorrect: expected match with Hugging Face model configuration, got None
✗ layer_norm_epsilon is incorrect: expected match with Hugging Face model configuration, got None
```

A sucessful test will show
```bash
✓ dataclass is correctly imported from dataclasses
✓ GPT2Config has the @dataclass decorator
✓ vocab_size is correct: 50257
✓ n_positions is correct: 1024
✓ n_embd is correct: 768
✓ n_layer is correct: 12
✓ n_head is correct: 12
✓ n_inner is correct: 3072
✓ layer_norm_epsilon is correct: 1e-05
```


**Reference**: `puzzles/config.py`

---

## Step 2: Implement Causal Masking (`causal.py`)

**Purpose**: Create attention masks to prevent the model from "seeing" future tokens.

**Key Concepts**:
- [**Causal masking**](https://docs.modular.com/glossary/ai/attention-mask/) is a technique that prevents tokens at a given position from attending to tokens at future positions by setting those attention scores to negative infinity before the softmax operation
- Causal masking ensures the model learns to predict the next token using only past context, which is essential for autoregressive text generation where the model must generate text one token at a time without "peeking" at future tokens
- MAX's [`@F.functional`](https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.functional) decorator converts a graph operation to support multiple tensor types
- MAX's [`F.band_part`](https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.band_part) copies a tensor setting everything outside a central band to zero

**Implementation Tasks**:
1. Import MAX tensor and functional modules
2. Create a `causal_mask` function decorated with `@F.functional`
3. Generate a mask tensor filled with `-inf`
4. Use `F.band_part` to keep only lower triangular portion (past tokens)
5. Return mask with shape `(sequence_length, sequence_length + num_tokens)`

**Reference**: `puzzles/causal.py`

---

## Step 3: Build Layer Normalization (`layernorm.py`)

**Purpose**: Implement layer normalization for stabilizing training.

**Key Concepts**:
- Layer normalization normalizes across the feature dimension
- Learnable `weight` (gamma) and `bias` (beta) parameters
- Use MAX's `F.layer_norm` functional operation

**Implementation Tasks**:
1. Create `LayerNorm` class inheriting from `Module`
2. Initialize `weight` and `bias` tensors
3. Implement `__call__` method using `F.layer_norm`
4. Pass epsilon for numerical stability

**Reference**: `puzzles/layernorm.py`

---

## Step 4: Implement MLP Layer (`mlp.py`)

**Purpose**: Build the feed-forward network within each transformer block.

**Key Concepts**:
- Two linear layers: expand to intermediate size, then project back
- Use GELU activation with tanh approximation
- Match Hugging Face naming: `c_fc` (expand), `c_proj` (project)

**Implementation Tasks**:
1. Create `GPT2MLP` class inheriting from `Module`
2. Initialize two `Linear` layers with bias
3. Implement forward pass: `Linear -> GELU -> Linear`
4. Use `F.gelu(x, approximate="tanh")` for activation

**Reference**: `puzzles/mlp.py`

---

## Step 5: Build Multi-Head Attention (`attention.py`)

**Purpose**: Implement the core self-attention mechanism.

**Key Concepts**:
- Multi-head attention: split embeddings across multiple "heads"
- Use single linear layer (`c_attn`) to project to Q, K, V simultaneously
- Apply causal masking to prevent attending to future tokens
- Scale attention scores by `1/sqrt(head_dim)`

**Implementation Tasks**:
1. Create `GPT2Attention` class with `c_attn` and `c_proj` linear layers
2. Implement `_split_heads`: reshape and transpose to `(batch, heads, seq, head_dim)`
3. Implement `_merge_heads`: reverse the split operation
4. Implement `_attn` method:
   - Compute attention scores: `Q @ K.T`
   - Scale by `1/sqrt(head_dim)`
   - Add causal mask
   - Apply softmax
   - Multiply by values: `attention @ V`
5. Implement `__call__`: project inputs, split heads, attend, merge heads, project output

**Reference**: `puzzles/attention.py`

---

## Step 6: Create Transformer Block (`block.py`)

**Purpose**: Combine attention and MLP with residual connections and layer normalization.

**Key Concepts**:
- Pre-normalization architecture (LayerNorm before attention/MLP)
- Residual connections around attention and MLP
- Two sub-layers: attention + MLP

**Implementation Tasks**:
1. Create `GPT2Block` class
2. Initialize two `LayerNorm` layers (`ln_1`, `ln_2`)
3. Initialize `GPT2Attention` and `GPT2MLP`
4. Implement forward pass:
   ```
   # Attention sub-layer
   residual = x
   x = ln_1(x)
   x = attention(x) + residual

   # MLP sub-layer
   residual = x
   x = ln_2(x)
   x = mlp(x) + residual
   ```

**Reference**: `puzzles/block.py`

---

## Step 7: Build Complete GPT-2 Model (`model.py`)

**Purpose**: Assemble the full transformer model with embeddings and language modeling head.

**Key Concepts**:
- Token embeddings (`wte`) and position embeddings (`wpe`)
- Stack multiple transformer blocks
- Final layer normalization
- Language modeling head (unembedding layer)

**Implementation Tasks**:
1. Create `GPTModel` class:
   - Initialize token and position embeddings
   - Create sequential stack of `n_layer` transformer blocks
   - Add final layer normalization
   - Implement forward pass: sum embeddings, apply blocks, normalize

2. Create `MaxGPT2Model` class:
   - Wrap `GPTModel` as `transformer`
   - Add `lm_head` linear layer (no bias)
   - Implement forward pass: get hidden states, project to vocab logits
   - Implement `load_huggingface_weights`:
     - Load state dict from Hugging Face model
     - Transpose Conv1D weights to match Linear format
     - Move to specified device

**Reference**: `puzzles/model.py`

---

## Step 8: Create Tokenizer Wrapper (`tokenizer.py`)

**Purpose**: Wrap Hugging Face tokenizer for MAX tensor compatibility.

**Key Concepts**:
- Use Hugging Face GPT2Tokenizer for text <-> token conversion
- Convert token lists to MAX tensors
- Handle device placement for tensors

**Implementation Tasks**:
1. Create `GPT2TokenizerWrapper` class
2. Initialize Hugging Face `GPT2Tokenizer`
3. Implement `encode`: text -> tensor of token IDs
4. Implement `decode`: tensor -> text using numpy interop
5. Add properties for `eos_token_id`, `pad_token_id`, `vocab_size`

**Reference**: `puzzles/tokenizer.py`

---

## Step 9: Implement Text Generator (`generator.py`)

**Purpose**: Generate text using the trained model with various sampling strategies.

**Key Concepts**:
- Autoregressive generation: predict next token, append, repeat
- Greedy decoding: always pick highest probability token
- Sampling: sample from probability distribution with temperature scaling
- Temperature: controls randomness (lower = more deterministic)

**Implementation Tasks**:
1. Create `TextGenerator` class
2. Implement `generate` method:
   - Encode prompt to tokens
   - Loop for `max_new_tokens` iterations:
     - Get model predictions (logits)
     - Extract logits for last position
     - If sampling:
       - Apply temperature scaling: `logits / temperature`
       - Convert to probabilities with softmax
       - Sample token from distribution
     - If greedy: select argmax token
     - Append new token to sequence
   - Decode and return final text

**Reference**: `puzzles/generator.py`

---

## Step 10: Create Demo Script (`demo.py`)

**Purpose**: Demonstrate the complete GPT-2 pipeline with text generation examples.

**Implementation Tasks**:
1. Load Hugging Face GPT-2 model for pretrained weights
2. Create MAX GPT-2 model instance
3. Transfer weights using `load_huggingface_weights`
4. Initialize tokenizer wrapper
5. Test forward pass with sample input
6. Create text generator
7. Run multiple generation examples:
   - Greedy generation (deterministic)
   - Sampling with moderate temperature
   - Creative writing with higher temperature

**Reference**: `puzzles/demo.py`

---

## Running the Implementation

Once all files are created:

```bash
python puzzles/demo.py
```

Expected output:
- Model initialization logs
- Forward pass test results
- Multiple text generation examples with different settings

---

## Key MAX Concepts Used

1. **Module System**: `max.nn.module_v3.Module` for building neural network layers
2. **Tensor Operations**: `max.experimental.tensor.Tensor` for array operations
3. **Functional API**: `max.experimental.functional` for operations like softmax, layer_norm, etc.
4. **Device Management**: `max.driver.Device` for GPU/CPU placement
5. **Graph Functions**: `@F.functional` decorator for JIT-compiled operations
6. **Weight Loading**: Interoperability with PyTorch/Hugging Face models

---

## Common Pitfalls to Avoid

1. **Weight Transposition**: Hugging Face uses Conv1D layers while MAX uses Linear - remember to transpose weights for `c_attn`, `c_proj`, `c_fc`
2. **Causal Masking**: Ensure mask is applied before softmax with correct shape
3. **Head Splitting**: Pay attention to dimension ordering when splitting/merging attention heads
4. **Device Consistency**: Keep all tensors on the same device throughout computation
5. **Temperature Scaling**: Apply before softmax, not after

---

## Next Steps

- Experiment with different sampling strategies (top-k, top-p)
- Implement KV-caching for faster generation
- Fine-tune on custom datasets
- Profile and optimize performance with MAX's compilation features
- Try larger GPT-2 variants (medium, large, xl)

---

## Resources

- [MAX Documentation](https://docs.modular.com/max)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
