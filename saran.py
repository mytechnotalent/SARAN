"""
SARAN: Shallow Auto-Regressive Attention Network (Multi-Layer Variant)

===============================================================================
THE 15-STEP SARAN ARCHITECTURE
===============================================================================

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
"""

import torch, torch.nn as nn, tiktoken, json, numpy as np
from torch.nn import functional as F

# ============== CONFIG ==============
B, T, C, L = 4, 512, 768, 12  # batch, context, embed, layers
grad_accum_steps = 16
max_iters, eval_interval, eval_iters = 50000, 1000, 100
lr, dropout, grad_clip = 6e-4, 0.0, 1.0
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.manual_seed(1337)

# ============== DATA ==============
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode, decode = lambda s: enc.encode(s), lambda l: enc.decode(list(l))

offsets = np.load("openwebtext_offsets.npy")
num_examples = len(offsets)
tokens_f = open("openwebtext_tokens.jsonl", "rb")


def get_batch(split):
    """Get a batch of data for training or validation."""
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


@torch.no_grad()
def eval_loss():
    """Estimate loss on train and validation sets."""
    model.eval()
    out = {
        s: torch.tensor([model(*get_batch(s))[1].item() for _ in range(eval_iters)]).mean()
        for s in ["train", "val"]
    }
    model.train()
    return out


# ============== MODEL ==============
# Step 13: RMSNorm - Pre-Layer Normalization (faster than LayerNorm)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# Steps 5-11: SARAN Attention Layer - Single Head (Key Innovation!)
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(C, 3 * C, bias=False)  # Steps 5, 6, 7: Fused Q, K, V
        self.proj = nn.Linear(C, C, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(T, T), diagonal=1).bool())

    def forward(self, x):
        _, t, _ = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # Step 8: Attention scores, Step 9: Causal mask, Step 10: Softmax, Step 11: Output
        w = (q @ k.transpose(-2, -1) * C**-0.5).masked_fill(self.mask[:t, :t], float("-inf"))
        return self.proj(F.softmax(w, dim=-1) @ v)


# Step 12: SARAN FFN - 2x Expansion (SARAN's Efficiency Innovation!)
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w2 = nn.Linear(C, C * 2, bias=False), nn.Linear(C * 2, C, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


# SARAN Block - Combines Steps 5-14 (Attn + FFN + Norm + Residual)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2, self.attn, self.ffn = RMSNorm(C), RMSNorm(C), Attn(), FFN()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Step 14: Residual around attention
        return x + self.ffn(self.ln2(x))  # Step 14: Residual around FFN


# SARAN Model - Complete 15-Step Architecture
class SARAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, C)  # Step 2: Token Embeddings
        self.pos = nn.Embedding(T, C)  # Step 3: Positional Encodings
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])  # Steps 5-14
        self.ln = RMSNorm(C)
        self.head = nn.Linear(C, vocab_size, bias=False)  # Step 15: Output Projection
        self.tok.weight = self.head.weight  # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, tgt=None):
        # Steps 1-4: Input tokens + embeddings
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=device))
        logits = self.head(self.ln(self.blocks(x)))  # Step 15
        return logits, (
            F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            if tgt is not None
            else None
        )

    def generate(self, idx, n, temp=0.8, top_k=40):
        for _ in range(n):
            logits = self(idx[:, -T:])[0][:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx


# ============== TRAIN ==============
print("=" * 70)
print("SARAN-Deep: Shallow Auto-Regressive Attention Network")
print("=" * 70)
print(f"  Embedding dimension:  {C}")
print(f"  Context length:       {T}")
print(f"  Number of layers:     {L}")
print(f"  Vocabulary size:      {vocab_size}")
print(f"  FFN expansion ratio:  2x (vs 4x in GPT)")
print(f"  Attention heads:      1 (single-head, vs 12 in GPT)")
print("=" * 70)

model = SARAN().to(device)
print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_iters, eta_min=lr / 10)

best_val_loss = float("inf")
for i in range(max_iters):
    if i % eval_interval == 0 or i == max_iters - 1:
        losses = eval_loss()
        print(f"step {i}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {sched.get_last_lr()[0]:.2e}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), "saran_best.pt")
            print(f"           -> New best model saved! (val_loss: {best_val_loss:.4f})")

    opt.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        loss = model(*get_batch("train"))[1]
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    sched.step()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config": {"vocab_size": vocab_size, "n_embd": C, "block_size": T, "n_layer": L},
        "iter": max_iters,
        "best_val_loss": best_val_loss,
    },
    "saran_pretrained.pt",
)
print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")

# ============== TEST ==============
model.load_state_dict(torch.load("saran_best.pt", map_location=device))
model.eval()
for prompt in ["Once upon a time", "The meaning of life is", "In a world where"]:
    print(f"\n[Prompt: '{prompt}']")
    print(decode(model.generate(torch.tensor([encode(prompt)], device=device), 100)[0].tolist()))

tokens_f.close()
