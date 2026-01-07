"""
SARAN-MLV Fine-Tuning Script

===============================================================================
FINE-TUNING SARAN ON INSTRUCTION DATA
===============================================================================

This script fine-tunes the pretrained SARAN-MLV model on the Stanford Alpaca
instruction-following dataset. The process transforms the base language model
into a chat-capable assistant.

Fine-Tuning Strategy:
    1. Load pretrained weights from saran_mlv_pretrained.pt
    2. Add dropout for regularization
    3. Train on instruction-response pairs
    4. Early stopping based on validation loss
    5. Save best checkpoint to saran_mlv_ft_best.pt

Dataset Format:
    User: <instruction>
    Assistant: <response>

Key Differences from Pretraining:
    - Lower learning rate (3e-5 vs 6e-4)
    - Dropout enabled (0.1)
    - Smaller batch with gradient accumulation
    - Early stopping with patience

===============================================================================
"""

import os
import ssl
import urllib.request

import tiktoken
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn import functional as F

# SSL and reproducibility
ssl._create_default_https_context = ssl._create_unverified_context
torch.manual_seed(1337)

# =============================================================================
# Configuration
# =============================================================================

# Model hyperparameters
block_size = 512  # Context length
n_embd = 768  # Embedding dimension
n_layer = 12  # Number of transformer layers
vocab_size = 50304  # Vocabulary size (padded for GPU efficiency)

# Training hyperparameters
batch_size = 4  # Batch size per step
grad_accum_steps = 4  # Gradient accumulation (effective batch = 16)
max_iters = 50000  # Maximum training iterations
eval_interval = 200  # Evaluate every N steps
learning_rate = 3e-5  # Learning rate (lower than pre-training)
dropout = 0.1  # Dropout probability for regularization
grad_clip = 1.0  # Gradient clipping threshold
patience = 5  # Early stopping patience (evals without improvement)

# Device and mixed precision
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
use_amp = device in ("mps", "cuda")
amp_dtype = torch.bfloat16 if use_amp else torch.float32

print(f"Device: {device} | AMP: {use_amp}")

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, disallowed_special=())
decode = lambda l: enc.decode(l, errors="replace")


# =============================================================================
# Dataset Preparation
# =============================================================================
finetune_file = "finetune_alpaca.txt"

# Download and process Alpaca dataset if needed
if not os.path.exists(finetune_file):
    print("Downloading Alpaca dataset...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "alpaca_data.json",
    )

    # Convert to conversation format
    alpaca = json.load(open("alpaca_data.json"))
    convs = []
    for item in alpaca:
        instr = item["instruction"].strip()
        inp = item.get("input", "").strip()
        out = item["output"].strip()
        user = f"{instr}\n{inp}" if inp else instr
        convs.append(f"User: {user}\nAssistant: {out}\n")

    open(finetune_file, "w").write("\n".join(convs))
    print(f"Saved {len(convs)} examples")

# Load and split dataset
data = torch.tensor(encode(open(finetune_file).read()), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"Train: {len(train_data):,} | Val: {len(val_data):,} tokens")


def get_batch(split_name):
    """
    Get a batch of data for training or validation.

    Args:
        split_name: "train" or "val"

    Returns:
        x: Input tokens (batch_size, block_size)
        y: Target tokens (batch_size, block_size)
    """
    d = train_data if split_name == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# =============================================================================
# RMSNorm - Root Mean Square Layer Normalization
# =============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Faster than LayerNorm, used in LLaMA and modern architectures.
    Computes: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim, eps=1e-6):
        """
        Initialize RMSNorm layer.

        Args:
            dim (int): The dimension of the input features to normalize.
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Apply RMS normalization to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized tensor of same shape as input.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# =============================================================================
# Attention Layer - Single Head with Dropout
# =============================================================================
class Attn(nn.Module):
    """
    SARAN's Single-Head Attention Layer with dropout.

    Unlike GPT-2 which uses 12 attention heads, SARAN uses a SINGLE head.
    Dropout is applied during training for regularization.
    """

    def __init__(self, dim, drop=0.0):
        """
        Initialize the single-head attention layer with dropout.

        Args:
            dim (int): Embedding dimension (also used for Q, K, V dimensions).
            drop (float, optional): Dropout probability applied during training.
                Defaults to 0.0.
        """
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = drop

    def forward(self, x):
        """
        Compute single-head causal self-attention with dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        q, k, v = self.qkv(x).split(x.size(-1), dim=-1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out)


# =============================================================================
# Feed-Forward Network - 2x Expansion with Dropout
# =============================================================================
class FFN(nn.Module):
    """
    SARAN's Feed-Forward Network with 2x expansion and dropout.

    GPT-2 uses 4x expansion (768 -> 3072 -> 768).
    SARAN uses 2x expansion (768 -> 1536 -> 768).
    """

    def __init__(self, dim, drop=0.0):
        """
        Initialize the feed-forward network with dropout.

        Args:
            dim (int): Embedding dimension. The hidden layer will be 2x this size.
            drop (float, optional): Dropout probability applied after the output
                projection. Defaults to 0.0.
        """
        super().__init__()
        hidden = dim * 2
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        """
        Apply feed-forward transformation with SiLU activation and dropout.

        Computes: Dropout(W2(SiLU(W1(x))))

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        return self.dropout(self.w2(F.silu(self.w1(x))))


# =============================================================================
# Transformer Block - Attention + FFN + Residuals
# =============================================================================
class Block(nn.Module):
    """
    One SARAN transformer block with dropout.

    Architecture (Pre-Norm style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, dim, drop=0.0):
        """
        Initialize a SARAN transformer block with dropout.

        Args:
            dim (int): Embedding dimension.
            drop (float, optional): Dropout probability for attention and FFN.
                Defaults to 0.0.
        """
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.attn = Attn(dim, drop)
        self.ffn = FFN(dim, drop)

    def forward(self, x):
        """
        Apply transformer block with pre-norm and residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# SARAN Model - Complete Architecture
# =============================================================================
class SARAN(nn.Module):
    """
    SARAN-MLV: Shallow Auto-Regressive Attention Network.

    Key Innovations:
        1. Single-head attention (not multi-head) - simpler, interpretable
        2. 2x FFN expansion (not 4x) - parameter efficient
        3. RMSNorm (not LayerNorm) - faster normalization
        4. Weight tying (embedding = output projection)

    Fine-tuning additions:
        - Dropout for regularization
        - Weight initialization
    """

    def __init__(self, vocab, embd, block, layers, drop=0.0):
        """
        Initialize the SARAN model for fine-tuning.

        Args:
            vocab (int): Size of the vocabulary (number of unique tokens).
            embd (int): Embedding dimension.
            block (int): Maximum sequence length (context window size).
            layers (int): Number of transformer blocks to stack.
            drop (float, optional): Dropout probability for regularization.
                Defaults to 0.0.
        """
        super().__init__()
        self.block_size = block

        # Token and position embeddings
        self.wte = nn.Embedding(vocab, embd)
        self.wpe = nn.Embedding(block, embd)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(embd, drop) for _ in range(layers)])

        # Final normalization and output projection
        self.ln_f = RMSNorm(embd)
        self.lm_head = nn.Linear(embd, vocab, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights with small random values.

        Applied recursively to all submodules via self.apply().
        Linear and Embedding layers are initialized with normal distribution.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass with optional loss computation.

        Args:
            idx: Input token indices (batch, seq_len)
            targets: Target tokens for loss computation

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        # Combine token and positional embeddings
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and output projection
        logits = self.lm_head(self.ln_f(x))

        # Compute loss if targets provided
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        )
        return logits, loss


# =============================================================================
# Model Initialization
# =============================================================================
model = SARAN(vocab_size, n_embd, block_size, n_layer, dropout).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Load pretrained weights
for path in ["saran_mlv_pretrained.pt", "saran_mlv_best.pt"]:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Loaded: {path}")
        break
else:
    print("WARNING: No pretrained weights!")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01
)


# =============================================================================
# Loss Estimation
# =============================================================================
@torch.no_grad()
def estimate_loss():
    """
    Estimate loss on train and validation sets.

    Returns:
        dict: {"train": avg_train_loss, "val": avg_val_loss}
    """
    model.eval()
    out = {}
    for split_name in ["train", "val"]:
        losses = [0.0] * 50
        for k in range(50):
            X, Y = get_batch(split_name)
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


# =============================================================================
# Training Loop
# =============================================================================
print("\nFine-tuning...")
best_val_loss = float("inf")
wait = 0

for it in range(max_iters):
    # Periodic evaluation
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:>5}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        # Save best model and check early stopping
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                "saran_mlv_ft_best.pt",
            )
            print(f"         -> saved (val={best_val_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stop at {it}")
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

print(f"\nDone! Best val loss: {best_val_loss:.4f}")
