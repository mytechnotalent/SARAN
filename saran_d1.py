"""
SARAN-Deep:  Shallow Auto-Regressive Attention Network (Multi-Layer Variant)

Extends the original 15-step architecture with optional stacking of attention layers.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# =============================================================================
# Reproducibility
# =============================================================================
torch.manual_seed(1337)

# =============================================================================
# Hyperparameters
# =============================================================================
batch_size = 4
grad_accum_steps = 16
block_size = 512
max_iters = 50000
eval_interval = 1000
learning_rate = 6e-4
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")
eval_iters = 100
n_embd = 768
n_layer = 12
dropout = 0.0
grad_clip = 1.0

# =============================================================================
# Dataset Loading (OpenWebText)
# =============================================================================
tokens_file = "openwebtext_tokens.jsonl"
offsets_file = "openwebtext_offsets.npy"
offsets = np.load(offsets_file)
num_examples = len(offsets)
tokens_f = open(tokens_file, "rb")

# =============================================================================
# Tokenizer
# =============================================================================
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(list(l))


# =============================================================================
# Data Loading
# =============================================================================
def get_batch(split):
    """Get a batch of data for training or validation."""
    split_idx = int(0.9 * num_examples)
    if split == "train":
        start_i, end_i = 0, split_idx
    else:
        start_i, end_i = split_idx, num_examples
    ex_count = end_i - start_i
    x_list = []
    y_list = []
    attempts = 0
    max_attempts = batch_size * 10
    while len(x_list) < batch_size and attempts < max_attempts:
        attempts += 1
        ex_id = start_i + np.random.randint(0, ex_count)
        tokens_f.seek(int(offsets[ex_id]))
        line = tokens_f.readline()
        toks = json.loads(line.decode("utf-8"))
        L = len(toks)
        if L <= block_size:
            continue
        start = np.random.randint(0, L - block_size)
        x_list.append(toks[start : start + block_size])
        y_list.append(toks[start + 1 : start + block_size + 1])
    x_np = np.array(x_list, dtype=np.int64)
    y_np = np.array(y_list, dtype=np.int64)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
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
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# =============================================================================
# RMSNorm - Essential for training stability
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# =============================================================================
# SARAN Attention Layer - Single Head (Key Innovation)
# =============================================================================
class SARANAttentionLayer(nn.Module):
    """Single-head attention with output projection."""

    def __init__(self, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.scale = n_embd**-0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool(),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return self.out_proj(attn @ v)


# =============================================================================
# SARAN FFN - 2x expansion (vs 4x in GPT) - More efficient
# =============================================================================
class SARANFFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = n_embd * 2
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


# =============================================================================
# SARAN Block - Pre-Norm style
# =============================================================================
class SARANBlock(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = SARANAttentionLayer(n_embd, block_size)
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SARANFFN(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# SARAN-Deep Model
# =============================================================================
class SARANDeep(nn.Module):
    """
    SARAN: Single-head Attention with 2x FFN expansion.
    Key claims: Simpler, more parameter-efficient than multi-head.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_layer, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [SARANBlock(n_embd, block_size) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        )
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat(
                (idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1
            )
        return idx


# =============================================================================
# Model Initialization
# =============================================================================
print("=" * 60)
print("SARAN-Deep: Single-Head Attention + 2x FFN")
print("=" * 60)

model = SARANDeep(vocab_size, n_embd, block_size, n_layer, dropout)
model = model.to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

# =============================================================================
# Optimizer
# =============================================================================
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, max_iters, eta_min=learning_rate / 10
)

# =============================================================================
# Training Loop
# =============================================================================
print("\nStarting training...")
best_val_loss = float("inf")

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.2e}"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), "saran_deep_best.pt")

    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

torch.save(
    {
        "model": model.state_dict(),
        "config": {"n_embd": n_embd, "n_layer": n_layer, "block_size": block_size},
    },
    "saran_pretrained.pt",
)
print(f"\nTraining complete! Best val: {best_val_loss:.4f}")

# Test generation
model.eval()
for prompt in ["Once upon a time", "The meaning of life is"]:
    tokens = torch.tensor([encode(prompt)], device=device)
    print(
        f"\n[{prompt}] {decode(model.generate(tokens, 100, temperature=0.8, top_k=40)[0].tolist())}"
    )

tokens_f.close()
