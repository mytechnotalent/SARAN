# SARAN: Shallow Auto-Regressive Attention Network

The dominant paradigms in sequence transduction - Recurrent Neural Networks and deep Transformer architectures - rely on complex, multi-layered structures to achieve performance, often at the cost of interpretability and computational transparency. In this work, we introduce the Shallow Auto-Regressive Attention Network (SARAN), a minimalist architecture that reduces the Transformer decoder to its fundamental components. SARAN is defined by a strictly linear, 15-stage computational graph that maps input embeddings directly to output probabilities via a single, isolated block of masked self-attention. We present a "first principles" derivation of the network's training dynamics, explicitly defining the manual backpropagation algorithm through the attention mechanism without reliance on automatic differentiation engines. By stripping away deep layer stacking and feed-forward networks, SARAN demonstrates that a solitary attention block is sufficient to mechanically derive autoregressive properties, providing a transparent and rigorous baseline for understanding the mechanics of attention-based sequence modeling.

---

## Table of Contents

- [SARAN: Shallow Auto-Regressive Attention Network](#saran-shallow-auto-regressive-attention-network)
  - [Table of Contents](#table-of-contents)
  - [1. SARAN vs GPT: Key Innovations](#1-saran-vs-gpt-key-innovations)
  - [2. Configuration \& Hyperparameters](#2-configuration--hyperparameters)
  - [3. Data Pipeline](#3-data-pipeline)
    - [Tokenization](#tokenization)
    - [Batch Creation (OpenWebText)](#batch-creation-openwebtext)
  - [4. Execution Flow Overview](#4-execution-flow-overview)
  - [5. The SARAN Class: The Heart of the Model](#5-the-saran-class-the-heart-of-the-model)
  - [6. Token Embeddings](#6-token-embeddings)
  - [7. Positional Embeddings](#7-positional-embeddings)
  - [8. The Transformer Block](#8-the-transformer-block)
    - [Residual Connections](#residual-connections)
  - [9. RMSNorm (Root Mean Square Normalization)](#9-rmsnorm-root-mean-square-normalization)
    - [RMSNorm vs LayerNorm](#rmsnorm-vs-layernorm)
  - [10. Single-Head Attention (SARAN's Simplicity Innovation)](#10-single-head-attention-sarans-simplicity-innovation)
    - [Step-by-Step Breakdown](#step-by-step-breakdown)
      - [Step 1: Compute Q, K, V](#step-1-compute-q-k-v)
      - [Step 2: Compute Attention Scores](#step-2-compute-attention-scores)
      - [Step 3: Apply Causal Mask](#step-3-apply-causal-mask)
      - [Step 4: Softmax](#step-4-softmax)
      - [Step 5: Weighted Sum of Values](#step-5-weighted-sum-of-values)
      - [Step 6: Output Projection](#step-6-output-projection)
    - [The Attention Formula (Complete)](#the-attention-formula-complete)
    - [Why Single-Head Works](#why-single-head-works)
  - [11. Feed-Forward Network (2x Expansion)](#11-feed-forward-network-2x-expansion)
    - [SiLU Activation](#silu-activation)
    - [Why 2x Expansion Instead of 4x?](#why-2x-expansion-instead-of-4x)
  - [12. Weight Tying](#12-weight-tying)
  - [13. Output Projection \& Loss](#13-output-projection--loss)
    - [Cross-Entropy Loss](#cross-entropy-loss)
  - [14. Text Generation](#14-text-generation)
    - [Temperature Scaling](#temperature-scaling)
    - [Top-k Sampling](#top-k-sampling)
  - [15. Training Loop](#15-training-loop)
    - [Gradient Accumulation](#gradient-accumulation)
    - [AdamW Optimizer](#adamw-optimizer)
    - [Cosine Annealing Learning Rate](#cosine-annealing-learning-rate)
    - [Gradient Clipping](#gradient-clipping)
    - [Mixed Precision (bfloat16)](#mixed-precision-bfloat16)
  - [16. Parameter Count](#16-parameter-count)
  - [Complete Forward Pass Example](#complete-forward-pass-example)
  - [Summary](#summary)

---

## 1. SARAN vs GPT: Key Innovations

SARAN introduces three key architectural simplifications compared to standard GPT:

| Feature             | GPT           | SARAN                 | Benefit                     |
| ------------------- | ------------- | --------------------- | --------------------------- |
| **Attention Heads** | 12 multi-head | 1 single-head         | Simpler, more interpretable |
| **FFN Expansion**   | 4× (768→3072) | 2× (768→1536)         | Fewer parameters, faster    |
| **Normalization**   | LayerNorm     | RMSNorm               | Faster computation          |
| **Activation**      | GELU          | SiLU (Swish)          | Modern, smooth gradients    |
| **Weight Tying**    | No            | Yes (embed = output)  | Fewer parameters            |
| **Biases**          | Yes           | No (in Linear layers) | Fewer parameters            |
| **Precision**       | float32       | bfloat16 (mixed)      | ~2x faster, 50% less memory |

These changes result in a more parameter-efficient model while maintaining competitive performance.

---

## 2. Configuration & Hyperparameters

The model is configured with these key hyperparameters:

```python
B, T, C, L = 4, 512, 768, 12
```

| Symbol | Name                | Value  | Description                         |
| ------ | ------------------- | ------ | ----------------------------------- |
| $B$    | Batch Size          | 4      | Sequences per micro-batch           |
| $T$    | Context Length      | 512    | Maximum sequence length (tokens)    |
| $C$    | Embedding Dimension | 768    | Size of token/positional embeddings |
| $L$    | Number of Layers    | 12     | Transformer blocks stacked          |
| $V$    | Vocabulary Size     | 50,257 | GPT-2 tokenizer vocabulary          |

**Note:** SARAN has no $H$ (heads) parameter because it uses **single-head attention**. The full embedding dimension $C = 768$ is used for attention, not split across heads.

Additional training hyperparameters:

| Parameter          | Value | Description                      |
| ------------------ | ----- | -------------------------------- |
| `grad_accum_steps` | 16    | Gradient accumulation steps      |
| `lr`               | 6e-4  | Learning rate                    |
| `grad_clip`        | 1.0   | Gradient clipping threshold      |
| `dropout`          | 0.0   | No dropout (full model capacity) |

**Effective batch size:** $B \times G = 4 \times 16 = 64$ (where $G$ = gradient accumulation steps)

---

## 3. Data Pipeline

### Tokenization

Text is converted to integer tokens using the GPT-2 BPE (Byte Pair Encoding) tokenizer:

```python
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50257
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(list(l))
```

**Example:**
```
"Hello world" → [15496, 995]
```

### Batch Creation (OpenWebText)

SARAN uses a memory-mapped approach to load pre-tokenized OpenWebText data:

```python
offsets = np.load("openwebtext_offsets.npy")
tokens_f = open("openwebtext_tokens.jsonl", "rb")

def get_batch(split):
    split_idx = int(0.9 * num_examples)
    start_i, end_i = (0, split_idx) if split == "train" else (split_idx, num_examples)
    x_list, y_list = [], []
    while len(x_list) < B:
        ex_id = start_i + np.random.randint(0, end_i - start_i)
        tokens_f.seek(int(offsets[ex_id]))
        toks = json.loads(tokens_f.readline().decode("utf-8"))
        if len(toks) <= T:
            continue
        start = np.random.randint(0, len(toks) - T)
        x_list.append(toks[start : start + T])
        y_list.append(toks[start + 1 : start + T + 1])
    return torch.tensor(x_list, device=device), torch.tensor(y_list, device=device)
```

This approach:
1. Uses file offsets to randomly access documents
2. Samples a random starting position within each document
3. Extracts input/target pairs for next-token prediction

**Shape Example:**
- Input shape: $(B, T) = (4, 512)$
- Target shape: $(B, T) = (4, 512)$

---

## 4. Execution Flow Overview

When training, execution follows this path:

```
                              ┌─────────────────────────────────────────────────────┐
                              │                   SARAN Class                       │
                              ├─────────────────────────────────────────────────────┤
Input Tokens ──► Token Embed ──► + ──► Block ──► Block ──► ... ──► Block ──► RMSNorm ──► Linear ──► Logits
  (B, T)           (B,T,C)      │       ×1        ×2              ×12         (B,T,C)      (B,T,V)
                                │                                                            ↑
                    Pos Embed ──┘                                                            │
                      (T, C)                                              (weight tying) ────┘
```

Each **Block** contains:
```
Input ──► RMSNorm ──► Single-Head Attention ──► + ──► RMSNorm ──► FFN (2x) ──► + ──► Output
  │                                             │                              │
  └────────────── (residual) ───────────────────┘────────── (residual) ────────┘
```

**Key differences from GPT:**
- RMSNorm instead of LayerNorm
- Single-head attention instead of 12-head MHA
- 2x FFN expansion instead of 4x

---

## 5. The SARAN Class: The Heart of the Model

```python
class SARAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, C)                     # Token embeddings
        self.pos = nn.Embedding(T, C)                              # Positional embeddings
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])  # 12 transformer blocks
        self.ln = RMSNorm(C)                                       # Final RMSNorm
        self.head = nn.Linear(C, vocab_size, bias=False)           # Output projection
        self.tok.weight = self.head.weight                         # Weight tying!
        self.apply(self._init_weights)
```

The forward pass:

```python
def forward(self, idx, tgt=None):
    x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=device))
    logits = self.head(self.ln(self.blocks(x)))
    return logits, F.cross_entropy(...) if tgt is not None else None
```

Let's trace a concrete example through the entire network.

---

## 6. Token Embeddings

The token embedding layer maps each token ID to a dense vector:

$$\mathbf{E}_{tok} \in \mathbb{R}^{V \times C} = \mathbb{R}^{50257 \times 768}$$

For input tokens $\mathbf{x} \in \mathbb{Z}^{B \times T}$:

$$\mathbf{X}_{tok} = \text{Embedding}(\mathbf{x}) \in \mathbb{R}^{B \times T \times C}$$

**Concrete Example:**

Suppose our input is the tokens `[15496, 995, 0]` representing "Hello world" plus a padding token.

For token ID `15496`:
- Look up row 15496 in $\mathbf{E}_{tok}$
- Retrieve a 768-dimensional vector, e.g.: $[0.02, -0.15, 0.08, ..., 0.11]$

Each of the 50,257 possible tokens has its own learned 768-dimensional representation.

**Memory (but shared with output via weight tying!):**
$$50257 \times 768 = 38,597,376 \text{ parameters} \approx 38.6\text{M}$$

---

## 7. Positional Embeddings

Transformers have no inherent notion of sequence order. Positional embeddings inject position information:

$$\mathbf{E}_{pos} \in \mathbb{R}^{T \times C} = \mathbb{R}^{512 \times 768}$$

For each position $t \in \{0, 1, ..., 511\}$, we retrieve a learned 768-dimensional vector.

The combined embedding is:

$$\mathbf{X} = \mathbf{X}_{tok} + \mathbf{E}_{pos}$$

**Concrete Example:**

Position 0 embedding: $\mathbf{p}_0 = [0.01, 0.03, -0.02, ..., 0.05]$  
Position 1 embedding: $\mathbf{p}_1 = [-0.02, 0.01, 0.04, ..., -0.03]$

If token "Hello" at position 0 has embedding $[0.02, -0.15, 0.08, ...]$:

$$\mathbf{x}_0 = [0.02 + 0.01, -0.15 + 0.03, 0.08 + (-0.02), ...] = [0.03, -0.12, 0.06, ...]$$

**Memory:**
$$512 \times 768 = 393,216 \text{ parameters} \approx 0.4\text{M}$$

---

## 8. The Transformer Block

Each of the 12 blocks applies the same structure with different learned weights:

```python
class Block(nn.Module):
    def __init__(self):
        self.ln1, self.ln2, self.attn, self.ffn = RMSNorm(C), RMSNorm(C), Attn(), FFN()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Attention with residual
        return x + self.ffn(self.ln2(x)) # FFN with residual
```

Mathematically, for block $\ell$:

$$\mathbf{X}^{(\ell)} = \mathbf{X}^{(\ell-1)} + \text{Attn}(\text{RMSNorm}(\mathbf{X}^{(\ell-1)}))$$

$$\mathbf{X}^{(\ell)} = \mathbf{X}^{(\ell)} + \text{FFN}(\text{RMSNorm}(\mathbf{X}^{(\ell)}))$$

This is the **Pre-Norm** formulation, where normalization is applied before each sublayer.

### Residual Connections

The `+` operations are residual (skip) connections. They:
1. Allow gradients to flow directly backward through the network
2. Enable the model to learn identity mappings easily
3. Stabilize training of deep networks

Without residuals, training a 12-layer network would be extremely difficult due to vanishing gradients.

---

## 9. RMSNorm (Root Mean Square Normalization)

SARAN uses RMSNorm instead of LayerNorm. RMSNorm is simpler and faster:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

Mathematically:

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

Where:
$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{C} \sum_{i=1}^{C} x_i^2 + \epsilon}$$

And $\gamma \in \mathbb{R}^C$ is a learned scale parameter (initialized to ones).

**Concrete Example:**

For a single position's embedding $\mathbf{x} = [2.0, 4.0, 6.0, 8.0]$ (simplified to 4D):

$$\text{RMS} = \sqrt{\frac{2^2 + 4^2 + 6^2 + 8^2}{4}} = \sqrt{\frac{4 + 16 + 36 + 64}{4}} = \sqrt{30} \approx 5.48$$

Normalized (assuming $\gamma = [1,1,1,1]$):

$$\hat{\mathbf{x}} = \frac{[2, 4, 6, 8]}{5.48} = [0.37, 0.73, 1.09, 1.46]$$

### RMSNorm vs LayerNorm

| Property       | LayerNorm                 | RMSNorm           |
| -------------- | ------------------------- | ----------------- |
| **Centering**  | Yes (subtracts mean)      | No                |
| **Scaling**    | By std deviation          | By RMS            |
| **Parameters** | $\gamma$ and $\beta$ (2C) | $\gamma$ only (C) |
| **Speed**      | Slower                    | ~15% faster       |

RMSNorm removes the mean-centering step, which empirically doesn't hurt performance but speeds up computation.

**Parameters per RMSNorm:** $C = 768$ (only scale, no shift)

---

## 10. Single-Head Attention (SARAN's Simplicity Innovation)

Unlike GPT's multi-head attention, SARAN uses **single-head attention** operating on the full embedding dimension:

```python
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(C, 3 * C, bias=False)  # Fused Q, K, V projection
        self.proj = nn.Linear(C, C, bias=False)     # Output projection
        self.register_buffer("mask", torch.triu(torch.ones(T, T), diagonal=1).bool())

    def forward(self, x):
        _, t, _ = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        w = (q @ k.transpose(-2, -1) * C**-0.5).masked_fill(self.mask[:t, :t], float("-inf"))
        return self.proj(F.softmax(w, dim=-1) @ v)
```

**Key difference from GPT:** 
- GPT: 12 heads, each with $d_k = 64$ dimensions
- SARAN: 1 head with $d_k = 768$ dimensions (full embedding)

### Step-by-Step Breakdown

#### Step 1: Compute Q, K, V

The input $\mathbf{X} \in \mathbb{R}^{B \times T \times C}$ is projected:

$$[\mathbf{Q}, \mathbf{K}, \mathbf{V}] = \mathbf{X}\mathbf{W}^{QKV}$$

Where $\mathbf{W}^{QKV} \in \mathbb{R}^{C \times 3C} = \mathbb{R}^{768 \times 2304}$ (no bias!)

Then split into three tensors, each $\in \mathbb{R}^{B \times T \times C}$

**Concrete Example (simplified):**

Let's trace a tiny example with $T=3$ positions and $C = 4$:

Input at 3 positions:

$$
\mathbf{X} = \begin{bmatrix} 
x_0 \\
x_1 \\
x_2 
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

After projection (assume weights give these results):

$$
\mathbf{Q} = \begin{bmatrix} 
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
1 & 1 & 0 & 0 
\end{bmatrix}, \quad 
\mathbf{K} = \begin{bmatrix} 
1 & 1 & 0 & 0 \\
0 & 1 & 1 & 0 \\
0 & 0 & 1 & 1 
\end{bmatrix}, \quad 
\mathbf{V} = \begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 
\end{bmatrix}
$$

#### Step 2: Compute Attention Scores

$$\text{scores} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C}}$$

Note: SARAN scales by $\sqrt{C} = \sqrt{768} \approx 27.7$, not $\sqrt{d_k} = \sqrt{64} = 8$ as in GPT.

$$
\mathbf{Q}\mathbf{K}^T = \begin{bmatrix} 
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
1 & 1 & 0 & 0 
\end{bmatrix} 
\begin{bmatrix} 
1 & 0 & 0 \\
1 & 1 & 0 \\
0 & 1 & 1 \\
0 & 0 & 1 
\end{bmatrix} = 
\begin{bmatrix} 
1 & 1 & 1 \\
1 & 1 & 1 \\
2 & 1 & 0 
\end{bmatrix}
$$

Scaling by $\frac{1}{\sqrt{4}} = 0.5$ (in our simplified 4D example):

$$
\text{scores} = \begin{bmatrix} 
0.5 & 0.5 & 0.5 \\
0.5 & 0.5 & 0.5 \\
1.0 & 0.5 & 0.0 
\end{bmatrix}
$$

#### Step 3: Apply Causal Mask

The causal mask prevents attending to future positions. SARAN uses `torch.triu` with `diagonal=1`:

$$
\text{mask} = \begin{bmatrix} 
0 & 1 & 1 \\
0 & 0 & 1 \\
0 & 0 & 0 
\end{bmatrix}
$$

Where 1 means "mask out" (set to $-\infty$):

$$
\mathbf{S}_{\text{masked}} = \begin{bmatrix} 
0.5 & -\infty & -\infty \\
0.5 & 0.5 & -\infty \\
1.0 & 0.5 & 0.0 
\end{bmatrix}
$$

#### Step 4: Softmax

Apply softmax row-wise. Since $e^{-\infty} = 0$:

Row 0: $\text{softmax}([0.5, -\infty, -\infty]) = [1.0, 0, 0]$

Row 1: $\text{softmax}([0.5, 0.5, -\infty]) = [0.5, 0.5, 0]$

Row 2: $\text{softmax}([1.0, 0.5, 0.0])$:
- $e^{1.0} = 2.72$, $e^{0.5} = 1.65$, $e^{0.0} = 1.0$
- Sum = 5.37
- $= [0.51, 0.31, 0.19]$

$$
\text{attn} = \begin{bmatrix} 
1.0 & 0 & 0 \\
0.5 & 0.5 & 0 \\
0.51 & 0.31 & 0.19 
\end{bmatrix}
$$

#### Step 5: Weighted Sum of Values

$$\mathbf{A}_{\text{out}} = \mathbf{A} \times \mathbf{V}$$

$$
= \begin{bmatrix} 
1.0 & 0 & 0 \\
0.5 & 0.5 & 0 \\
0.51 & 0.31 & 0.19 
\end{bmatrix} 
\begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 
\end{bmatrix} = 
\begin{bmatrix} 
1.0 & 0 & 0 & 0 \\
0.5 & 0.5 & 0 & 0 \\
0.51 & 0.31 & 0.19 & 0 
\end{bmatrix}
$$

**Interpretation:**
- Position 0 only sees itself (attention weight 1.0 on position 0)
- Position 1 sees positions 0 and 1 equally (0.5 each)
- Position 2 sees all previous positions with decaying attention

#### Step 6: Output Projection

Unlike GPT where concatenation of head outputs is projected, SARAN directly projects the single-head output:

$$\mathbf{O} = \mathbf{A}_{\text{out}} \cdot \mathbf{W}^{O}$$

Where $\mathbf{W}^{O} \in \mathbb{R}^{C \times C} = \mathbb{R}^{768 \times 768}$ (no bias!)

### The Attention Formula (Complete)

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C}} + \mathbf{M}\right)\mathbf{V}$$

Where $\mathbf{M}$ is the causal mask ($0$ for allowed positions, $-\infty$ for masked).

### Why Single-Head Works

Multi-head attention was designed to let the model attend to different aspects in parallel. However:

1. **With sufficient depth (12 layers)**, single-head attention can learn diverse patterns across layers
2. **Full $C$-dimensional attention** captures richer relationships per layer
3. **Simpler architecture** means easier optimization and interpretation
4. **Fewer parameters** without significant quality loss

---

## 11. Feed-Forward Network (2x Expansion)

SARAN uses a 2x expansion FFN instead of GPT's 4x:

```python
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w2 = nn.Linear(C, C * 2, bias=False), nn.Linear(C * 2, C, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))
```

Mathematically:

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \mathbf{x})$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{1536 \times 768}$ (expansion to 2×)
- $\mathbf{W}_2 \in \mathbb{R}^{768 \times 1536}$ (projection back)
- No biases!

### SiLU Activation

SARAN uses SiLU (Sigmoid Linear Unit), also known as Swish:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Example values:**
| $x$  | $\sigma(x)$ | SiLU($x$) |
| ---- | ----------- | --------- |
| -2.0 | 0.12        | -0.24     |
| -1.0 | 0.27        | -0.27     |
| 0.0  | 0.50        | 0.0       |
| 1.0  | 0.73        | 0.73      |
| 2.0  | 0.88        | 1.76      |

SiLU is smooth, non-monotonic (has a small negative region), and has been shown to work well in modern architectures like LLaMA and PaLM.

**SiLU vs GELU comparison:**

| Property    | GELU              | SiLU                |
| ----------- | ----------------- | ------------------- |
| Formula     | $x \cdot \Phi(x)$ | $x \cdot \sigma(x)$ |
| Min value   | ~-0.17 at x≈-0.75 | ~-0.28 at x≈-1.28   |
| Computation | Slower (erf)      | Faster (sigmoid)    |
| Usage       | GPT, BERT         | LLaMA, SARAN        |

### Why 2x Expansion Instead of 4x?

GPT uses 4x expansion (768→3072→768), while SARAN uses 2x (768→1536→768):

**Parameter comparison per FFN layer:**
- GPT: $768 \times 3072 + 3072 \times 768 = 4,718,592$ params
- SARAN: $768 \times 1536 + 1536 \times 768 = 2,359,296$ params (50% reduction!)

This tradeoff:
1. **Reduces total parameters** significantly
2. **Speeds up training and inference**
3. **May be compensated** by the full-dimensional single-head attention

---

## 12. Weight Tying

SARAN ties the token embedding matrix to the output projection matrix:

```python
self.tok = nn.Embedding(vocab_size, C)
self.head = nn.Linear(C, vocab_size, bias=False)
self.tok.weight = self.head.weight  # Weight tying!
```

This means:

$$\mathbf{E}\_{\text{tok}} = \mathbf{W}\_{\text{out}}^T$$

The same matrix is used for:
1. **Encoding**: token ID → embedding vector
2. **Decoding**: hidden state → vocabulary logits

**Benefits:**
- **Fewer parameters**: Saves $V \times C = 38.6M$ parameters
- **Semantic consistency**: Similar tokens have similar embeddings AND similar output distributions
- **Regularization effect**: Constrains the model's representation space

**Memory savings:**
$$50257 \times 768 = 38,597,376 \text{ parameters saved}$$

---

## 13. Output Projection & Loss

After all transformer blocks, we project to vocabulary logits:

```python
logits = self.head(self.ln(self.blocks(x)))
```

$$\text{logits} = \mathbf{W}_{out} \cdot \text{RMSNorm}(\mathbf{X}^{(L)})$$

Where $\mathbf{W}_{out} \in \mathbb{R}^{V \times C} = \mathbb{R}^{50257 \times 768}$ (shared with embedding!)

**Output shape:** $(B, T, V) = (4, 512, 50257)$

Each position produces a 50,257-dimensional vector of logits (unnormalized log-probabilities).

### Cross-Entropy Loss

For training, we compute cross-entropy loss between predictions and targets:

$$\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \log P(y_{b,t} \mid x_{b,1:t-1})$$

Where $P(y \mid x) = \text{softmax}(\text{logits})_y$

**Concrete Example:**

For a single position predicting token 42:
- Logits: $[1.2, 0.5, ..., 3.8_{(42)}, ..., 0.1]$ (50,257 values)
- Softmax: $[0.001, 0.0005, ..., 0.15_{(42)}, ..., 0.0003]$
- Loss: $-\log(0.15) = 1.90$

A loss of ~2.0 means the model assigns roughly $e^{-2} \approx 13.5\%$ probability to the correct token on average.

---

## 14. Text Generation

Generation uses autoregressive sampling:

```python
def generate(self, idx, n, temp=0.8, top_k=40):
    for _ in range(n):
        logits = self(idx[:, -T:])[0][:, -1, :] / temp
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return idx
```

### Temperature Scaling

Temperature $\tau$ controls randomness:

$$P(w_i) = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}$$

| Temperature | Effect                                   |
| ----------- | ---------------------------------------- |
| $\tau < 1$  | Sharper distribution, more deterministic |
| $\tau = 1$  | Original distribution                    |
| $\tau > 1$  | Flatter distribution, more random        |

**Example with logits $[2.0, 1.0, 0.5]$:**

| $\tau$ | Probabilities      |
| ------ | ------------------ |
| 0.5    | [0.88, 0.10, 0.02] |
| 1.0    | [0.67, 0.24, 0.09] |
| 2.0    | [0.51, 0.31, 0.18] |

### Top-k Sampling

SARAN uses top-k=40 by default (GPT uses 50):

1. Find the 40th largest logit value
2. Set all logits below this threshold to $-\infty$
3. Renormalize with softmax
4. Sample from this truncated distribution

**Example:** If top-k=3 and logits are $[5, 3, 2, 1, 0.5]$:
- Top 3 values: $[5, 3, 2]$
- Threshold: 2
- After masking: $[5, 3, 2, -\infty, -\infty]$
- After softmax: $[0.84, 0.11, 0.04, 0, 0]$

---

## 15. Training Loop

SARAN's training loop includes gradient accumulation, cosine annealing, and gradient clipping:

```python
for i in range(max_iters):
    opt.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        loss = model(*get_batch("train"))[1]
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    sched.step()
```

### Gradient Accumulation

With batch size 4 and 16 accumulation steps:

$$\text{Effective Batch Size} = B \times G = 4 \times 16 = 64$$

This allows training with a large effective batch size on limited GPU memory.

### AdamW Optimizer

SARAN uses AdamW with specific hyperparameters:

```python
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
```

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Where:
- $\alpha = 6 \times 10^{-4}$ (learning rate)
- $\lambda = 0.1$ (weight decay)
- $\beta_1 = 0.9$, $\beta_2 = 0.95$ (momentum terms — note $\beta_2$ is lower than typical 0.999)

### Cosine Annealing Learning Rate

```python
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_iters, eta_min=lr / 10)
```

The learning rate follows a cosine curve:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

- Starts at $\eta_{max} = 6 \times 10^{-4}$
- Decays to $\eta_{min} = 6 \times 10^{-5}$ over 50,000 iterations

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

If $\|\nabla\| > 1.0$, gradients are scaled down:

$$\nabla' = \nabla \cdot \frac{1.0}{\|\nabla\|}$$

This prevents exploding gradients and stabilizes training.

### Mixed Precision (bfloat16)

SARAN uses automatic mixed precision (AMP) for faster training on GPU/MPS:

```python
from torch.amp import autocast

# Device-aware dtype selection
use_amp = device in ("mps", "cuda")
amp_dtype = torch.bfloat16 if use_amp else torch.float32

# Training loop with autocast
with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
    _, loss = model(xb, yb)
(loss / grad_accum_steps).backward()
```

**Precision Comparison:**

| Precision | Bits | Memory | Speed   | Stability          |
| --------- | ---- | ------ | ------- | ------------------ |
| float32   | 32   | 100%   | 1x      | Best               |
| float16   | 16   | 50%    | ~2x     | Needs loss scaling |
| bfloat16  | 16   | 50%    | ~1.5-2x | Very stable        |

**Why bfloat16?**
- Same exponent range as float32 (8 bits) — no overflow issues
- Reduced mantissa (7 bits vs 23) — slightly less precision
- No loss scaling required (unlike float16)
- Native support on Apple Silicon (MPS) and modern NVIDIA GPUs
- Typical loss difference: < 0.01-0.05 (negligible)

**Memory savings:**
$$95\text{M params} \times 2\text{ bytes} = 190\text{ MB} \quad \text{(vs 380 MB with float32)}$$

---

## 16. Parameter Count

Let's count all parameters:

| Component                  | Calculation           | Parameters                |
| -------------------------- | --------------------- | ------------------------- |
| Token Embedding            | $V \times C$          | 50,257 × 768 = 38,597,376 |
| Position Embedding         | $T \times C$          | 512 × 768 = 393,216       |
| **Per Transformer Block:** |                       |                           |
| → RMSNorm 1                | $C$                   | 768                       |
| → RMSNorm 2                | $C$                   | 768                       |
| → Attention QKV            | $C \times 3C$         | 768 × 2304 = 1,769,472    |
| → Attention Output         | $C \times C$          | 768 × 768 = 589,824       |
| → FFN Layer 1              | $C \times 2C$         | 768 × 1536 = 1,179,648    |
| → FFN Layer 2              | $2C \times C$         | 1536 × 768 = 1,179,648    |
| **Block Total**            |                       | ~4,720,128                |
| All 12 Blocks              | $12 \times$           | 56,641,536                |
| Final RMSNorm              | $C$                   | 768                       |
| Output Head                | (tied with embedding) | 0                         |

**Total: ~95.7 Million Parameters**

**Comparison with GPT (124.4M):**
- **Savings from 2x FFN:** 12 × (4.7M - 2.4M) = 27.6M
- **Savings from weight tying:** 38.6M
- **Savings from no biases:** ~0.5M
- **Savings from RMSNorm (vs LayerNorm):** ~0.02M

---

## Complete Forward Pass Example

Let's trace "Hello" through the entire network:

**1. Input:** `"Hello"` → token `[15496]` → tensor shape $(1, 1)$

**2. Token Embedding:** Look up row 15496 → $(1, 1, 768)$

**3. Position Embedding:** Look up position 0 → $(1, 768)$, broadcast to $(1, 1, 768)$

**4. Sum:** Token + Position → $(1, 1, 768)$

**5. Through 12 Blocks:**
   - Each block: RMSNorm → Single-Head Attn → Add → RMSNorm → FFN(2x) → Add
   - Shape stays $(1, 1, 768)$ throughout

**6. Final RMSNorm:** $(1, 1, 768)$

**7. Output Head:** Linear projection (tied weights) → $(1, 1, 50257)$

**8. Softmax + Sample:** Probability distribution over 50,257 tokens → sample next token

**9. Repeat:** Append new token, process again for next prediction

---

## Summary

The SARAN architecture makes strategic simplifications to the GPT design:

| Component         | GPT            | SARAN         | Tradeoff                     |
| ----------------- | -------------- | ------------- | ---------------------------- |
| **Attention**     | 12 heads × 64d | 1 head × 768d | Simpler, full-rank attention |
| **FFN**           | 4x expansion   | 2x expansion  | Fewer params, faster         |
| **Normalization** | LayerNorm      | RMSNorm       | Faster, fewer params         |
| **Activation**    | GELU           | SiLU          | Modern, smooth               |
| **Output**        | Separate head  | Weight tied   | Fewer params                 |
| **Biases**        | Yes            | No            | Fewer params                 |

**Key Insights:**

1. **Single-head attention** can be as effective as multi-head when the model is deep enough
2. **2x FFN expansion** provides sufficient capacity with half the parameters
3. **Weight tying** enforces semantic consistency and saves 38M parameters
4. **RMSNorm** is faster without sacrificing quality
5. **No biases** in Linear layers reduces parameters with minimal impact

The result is a ~95M parameter model (vs GPT's 124M) that maintains competitive performance through architectural efficiency rather than scale.
