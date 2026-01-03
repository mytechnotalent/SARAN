"""
SARAN-MLV: Shallow Auto-Regressive Attention Network (Fine-Tuning Variant)

===============================================================================
FINE-TUNING THE 15-STEP SARAN ARCHITECTURE ON CoQA
===============================================================================

This script fine-tunes a pre-trained SARAN model on the CoQA conversational
question-answering dataset. The architecture remains identical to saran_mlv.py.

Step 1:  Input Tokens
        - Raw token indices from the vocabulary

Step 2:  Token Embeddings (W_embed)
        - Map each token to a dense vector representation
        - Shape: (batch, seq_len) -> (batch, seq_len, n_embd)

Step 3:  Positional Encodings (W_pos)
        - Learned positional embeddings for each position
        - Shape: (seq_len,) -> (seq_len, n_embd)

Step 4:  Embedding Summation
        - Combine token and positional embeddings
        - x = token_emb + pos_emb

Step 5:  Query Projection (W_q)
        - Project embeddings to query space
        - Q = x @ W_q

Step 6:  Key Projection (W_k)
        - Project embeddings to key space
        - K = x @ W_k

Step 7:  Value Projection (W_v)
        - Project embeddings to value space
        - V = x @ W_v

Step 8:  Attention Score Calculation
        - Compute scaled dot-product attention scores
        - scores = (Q @ K^T) / sqrt(d_k)

Step 9:  Causal Masking
        - Apply causal mask to prevent attending to future tokens
        - scores = mask_fill(scores, mask, -inf)

Step 10: Softmax
        - Normalize attention scores to probabilities
        - attn = softmax(scores, dim=-1)

Step 11: Attention Output Calculation
        - Weighted sum of values
        - attn_out = attn @ V

Step 12: Feed-Forward Network (SARAN uses 2x expansion, not 4x)
        - FFN(x) = W2(SiLU(W1(x)))
        - This is SARAN's efficiency innovation

Step 13: Layer Normalization (RMSNorm)
        - Normalize before each sub-layer (Pre-Norm)

Step 14: Residual Connections
        - x = x + sublayer(norm(x))

Step 15: Output Projection (W_out) + Softmax
        - Project to vocabulary size
        - logits = x @ W_out
        - probs = softmax(logits)

===============================================================================
SARAN KEY INNOVATIONS:
===============================================================================
1. Single-Head Attention (not multi-head) - simpler, more interpretable
2. 2x FFN Expansion (not 4x) - more parameter efficient
3. RMSNorm (not LayerNorm) - faster, equally effective

===============================================================================
"""

import json
import os
import ssl
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast
import tiktoken

ssl._create_default_https_context = ssl._create_unverified_context

# =============================================================================
# Reproducibility
# =============================================================================
torch.manual_seed(1337)

# =============================================================================
# Hyperparameters (Fine-Tuning)
# =============================================================================
batch_size = 4
grad_accum_steps = 4
block_size = 512
max_iters = 2000
eval_interval = 100
learning_rate = 5e-5  # Lower learning rate for fine-tuning
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# Mixed precision dtype (bfloat16 for MPS/CUDA, float32 for CPU)
use_amp = device in ("mps", "cuda")
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"Using mixed precision: {use_amp} ({amp_dtype})")

eval_iters = 50
n_embd = 768
n_layer = 12
dropout = 0.1  # Add dropout for fine-tuning regularization
grad_clip = 1.0
patience = 5  # Early stopping patience

# =============================================================================
# Tokenizer
# =============================================================================
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, disallowed_special=())
decode = lambda l: enc.decode(list(l))

# =============================================================================
# Dataset Loading (CoQA Conversational Q&A)
# =============================================================================
finetune_file = "finetune.txt"

if not os.path.exists(finetune_file):
    print("Downloading CoQA dataset...")
    urllib.request.urlretrieve(
        "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json", "coqa.json"
    )
    coqa = json.load(open("coqa.json"))
    conversations = []
    for item in coqa["data"][:1000]:
        conv = f"Context: {item['story'][:400]}\n"
        for q, a in zip(item["questions"], item["answers"]):
            conv += f"User: {q['input_text']}\nAssistant: {a['input_text']}\n"
        conversations.append(conv)
    open(finetune_file, "w").write("\n\n".join(conversations))
    print(f"Saved {len(conversations)} Q&A conversations")

# Tokenize the fine-tuning data
data = torch.tensor(encode(open(finetune_file).read()), dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


# =============================================================================
# Data Loading
# =============================================================================
def get_batch(split):
    """Get a batch of data for training or validation."""
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# =============================================================================
# Step 13: RMSNorm - Pre-Layer Normalization (faster than LayerNorm)
# =============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Faster than LayerNorm, used in LLaMA and modern architectures.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# =============================================================================
# Steps 5-11: SARAN Attention Layer - Single Head (Key Innovation!)
# =============================================================================
class SARANAttentionLayer(nn.Module):
    """
    SARAN's Single-Head Attention Layer.

    Unlike GPT-2 which uses 12 attention heads, SARAN uses a SINGLE head.
    This is simpler, more interpretable, and we hypothesize equally effective.

    Implements Steps 5-11:
        Step 5:  Query Projection (W_q)
        Step 6:  Key Projection (W_k)
        Step 7:  Value Projection (W_v)
        Step 8:  Attention Score Calculation - scores = (Q @ K^T) / sqrt(d_k)
        Step 9:  Causal Masking - prevent attending to future tokens
        Step 10: Softmax - normalize to probabilities
        Step 11: Attention Output - weighted sum of values
    """

    def __init__(self, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.scale = n_embd**-0.5  # 1/sqrt(d_k) for scaled dot-product

        # Steps 5, 6, 7: Fused Q, K, V projection (more efficient)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # Output projection after attention
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Step 9: Causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool(),
        )

        # Dropout for regularization during fine-tuning
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Steps 5, 6, 7: Q, K, V Projections
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # Step 8: Attention Score Calculation
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Step 9: Causal Masking
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # Step 10: Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Step 11: Attention Output
        return self.out_proj(attn @ v)


# =============================================================================
# Step 12: SARAN FFN - 2x Expansion (SARAN's Efficiency Innovation!)
# =============================================================================
class SARANFFN(nn.Module):
    """
    SARAN's Feed-Forward Network with 2x expansion.

    GPT-2 uses 4x expansion (768 -> 3072 -> 768).
    SARAN uses 2x expansion (768 -> 1536 -> 768).

    This is more parameter-efficient while maintaining quality.
    Uses SiLU (Swish) activation instead of GELU.

    FFN(x) = W2(SiLU(W1(x)))
    """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = n_embd * 2  # 2x expansion (SARAN innovation, vs 4x in GPT)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))))


# =============================================================================
# SARAN Block - Combines Steps 5-14 (Attn + FFN + Norm + Residual)
# =============================================================================
class SARANBlock(nn.Module):
    """
    One SARAN transformer block.

    Architecture (Pre-Norm style):
        x = x + Attention(RMSNorm(x))   # Steps 5-11, 13, 14
        x = x + FFN(RMSNorm(x))         # Steps 12, 13, 14
    """

    def __init__(self, n_embd, block_size, dropout=0.0):
        super().__init__()
        # Step 13: Pre-normalization
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

        # Steps 5-11: Single-head attention
        self.attn = SARANAttentionLayer(n_embd, block_size, dropout)

        # Step 12: Feed-forward network (2x expansion)
        self.ffn = SARANFFN(n_embd, dropout)

    def forward(self, x):
        # Step 14: Residual connection around attention
        x = x + self.attn(self.ln1(x))

        # Step 14: Residual connection around FFN
        x = x + self.ffn(self.ln2(x))

        return x


# =============================================================================
# SARAN-MLV Model - Complete 15-Step Architecture
# =============================================================================
class SARANMLV(nn.Module):
    """
    SARAN-MLV: Shallow Auto-Regressive Attention Network

    Complete implementation of the 15-step architecture:
        Steps 1-4:   Input processing (embeddings)
        Steps 5-14:  Repeated N times (n_layer SARAN blocks)
        Step 15:     Output projection to vocabulary

    Key Innovations:
        1. Single-head attention (not multi-head) - simpler, interpretable
        2. 2x FFN expansion (not 4x) - parameter efficient
        3. RMSNorm (not LayerNorm) - faster normalization
        4. Weight tying (embedding = output projection)
    """

    def __init__(self, vocab_size, n_embd, block_size, n_layer, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer

        # Step 2: Token Embeddings (W_embed)
        self.wte = nn.Embedding(vocab_size, n_embd)

        # Step 3: Positional Encodings (W_pos) - learned
        self.wpe = nn.Embedding(block_size, n_embd)

        # Steps 5-14: Stack of SARAN blocks (repeated n_layer times)
        self.blocks = nn.ModuleList(
            [SARANBlock(n_embd, block_size, dropout) for _ in range(n_layer)]
        )

        # Final normalization before output
        self.ln_f = RMSNorm(n_embd)

        # Step 15: Output Projection (W_out)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass implementing all 15 steps.

        Args:
            idx: Input token indices (batch, seq_len) - Step 1
            targets: Target tokens for loss computation

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape

        # Step 1: Input Tokens (idx is the input)
        # Step 2: Token Embeddings
        # Step 3: Positional Encodings
        # Step 4: Embedding Summation
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))

        # Steps 5-14: Apply each SARAN block
        for block in self.blocks:
            x = block(x)

        # Step 15: Final norm + Output Projection
        logits = self.lm_head(self.ln_f(x))

        # Compute loss if targets provided
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        )
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            idx: Starting token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            logits, _ = self(idx[:, -self.block_size :])

            # Get last position logits and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample and append
            idx = torch.cat(
                (idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1
            )
        return idx


# =============================================================================
# Model Initialization
# =============================================================================
print("=" * 70)
print("SARAN-MLV: Fine-Tuning on CoQA")
print("=" * 70)
print(f"  Embedding dimension:  {n_embd}")
print(f"  Context length:       {block_size}")
print(f"  Number of layers:     {n_layer}")
print(f"  Vocabulary size:      {vocab_size}")
print(f"  FFN expansion ratio:  2x (vs 4x in GPT)")
print(f"  Attention heads:      1 (single-head, vs 12 in GPT)")
print(f"  Dropout:              {dropout}")
print(f"  Learning rate:        {learning_rate}")
print("=" * 70)

model = SARANMLV(vocab_size, n_embd, block_size, n_layer, dropout)
model = model.to(device)

# =============================================================================
# Load Pre-trained Weights
# =============================================================================
pretrained_path = "saran_mlv_pretrained.pt"
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pre-trained weights from {pretrained_path}")
        if "best_val_loss" in checkpoint:
            print(f"  Pre-training best val loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded pre-trained weights from {pretrained_path}")
else:
    # Try loading from best model
    best_path = "saran_mlv_best.pt"
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print(f"Loaded pre-trained weights from {best_path}")
    else:
        print("WARNING: No pre-trained weights found! Training from scratch.")

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params / 1e6:.2f}M")
print("=" * 70)

# =============================================================================
# Optimizer (No Scheduler for Fine-Tuning - constant low LR)
# =============================================================================
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01
)

# =============================================================================
# Fine-Tuning Loop with Early Stopping
# =============================================================================
print("\nStarting fine-tuning...")
print("-" * 70)

best_val_loss = float("inf")
wait = 0

for it in range(max_iters):
    # Evaluation
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {it:>6d}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

        # Save best model and check early stopping
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            wait = 0
            torch.save(model.state_dict(), "saran_mlv_ft_best.pt")
            print(
                f"           -> New best model saved! (val_loss: {best_val_loss:.4f})"
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at step {it} (patience={patience})")
                break

    # Training step with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        xb, yb = get_batch("train")
        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(xb, yb)
        (loss / grad_accum_steps).backward()

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# =============================================================================
# Load Best Model
# =============================================================================
model.load_state_dict(
    torch.load("saran_mlv_ft_best.pt", map_location=device, weights_only=True)
)
print("\nLoaded best fine-tuned model")

# =============================================================================
# Save Final Checkpoint
# =============================================================================
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "block_size": block_size,
            "n_layer": n_layer,
            "dropout": dropout,
        },
        "iter": it,
        "best_val_loss": best_val_loss,
    },
    "saran_mlv_finetuned.pt",
)

print("\n" + "=" * 70)
print("Fine-tuning complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("=" * 70)

# =============================================================================
# Test Generation
# =============================================================================
print("\nGenerating sample responses...")
print("-" * 70)

model.eval()

prompts = [
    "User: What is AI?\nAssistant:",
    "User: Tell me about machine learning.\nAssistant:",
    "User: How does a neural network work?\nAssistant:",
]

for prompt in prompts:
    print(f"\n[Prompt: '{prompt}']")
    tokens = torch.tensor([encode(prompt)], device=device)
    generated = model.generate(tokens, max_new_tokens=100, temperature=0.8, top_k=40)
    print(decode(generated[0].tolist()))

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
