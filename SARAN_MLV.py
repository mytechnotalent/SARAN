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
batch_size = 8
grad_accum_steps = 8
block_size = 256
max_iters = 20000
eval_interval = 500
learning_rate = 3e-4
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda. is_available() else "cpu"
)
print(f"Using device: {device}")
eval_iters = 200
n_embd = 640
n_layer = 3  # NEW: Number of attention layers (original SARAN = 1)
dropout = 0.1
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
vocab_size = enc. n_vocab
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
        line = tokens_f. readline()
        toks = json.loads(line. decode("utf-8"))
        L = len(toks)
        if L <= block_size: 
            continue
        start = np.random.randint(0, L - block_size)
        x_list.append(toks[start:  start + block_size])
        y_list.append(toks[start + 1: start + block_size + 1])
    x_np = np.array(x_list, dtype=np.int64)
    y_np = np.array(y_list, dtype=np.int64)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets."""
    out = {}
    model. eval()
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
# SARAN Attention Layer (Steps 5-11 from original architecture)
# =============================================================================
class SARANAttentionLayer(nn. Module):
    """
    Single SARAN attention layer implementing steps 5-11:
    5. Query Projection
    6. Key Projection
    7. Value Projection
    8. Attention Score Calculation
    9. Causal Masking
    10. Softmax
    11. Attention Output Calculation
    """
    
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.scale = n_embd ** 0.5
        
        # Steps 5, 6, 7:  Q, K, V Projections
        self.w_q = nn. Linear(n_embd, n_embd, bias=False)
        self.w_k = nn.Linear(n_embd, n_embd, bias=False)
        self.w_v = nn.Linear(n_embd, n_embd, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Step 5: Query Projection
        q = self.w_q(x)
        
        # Step 6: Key Projection
        k = self. w_k(x)
        
        # Step 7: Value Projection
        v = self.w_v(x)
        
        # Step 8: Attention Score Calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Step 9: Causal Masking
        scores = scores.masked_fill(self.causal_mask[: T, :T], float('-inf'))
        
        # Step 10: Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Step 11: Attention Output Calculation
        attn_out = torch.matmul(attn, v)
        attn_out = self.resid_dropout(attn_out)
        
        # Residual connection (helps gradient flow through layers)
        return x + attn_out


# =============================================================================
# SARAN-Deep Model
# =============================================================================
class SARANDeep(nn.Module):
    """
    SARAN with multiple stacked attention layers. 
    
    Architecture:
    - Steps 1-4: Input processing (embeddings)
    - Steps 5-11: Repeated N times (n_layer attention layers)
    - Steps 12-15: Output processing (projection to vocab)
    """
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self. n_layer = n_layer
        
        # Step 2: Token Embeddings
        self.w_embed = nn.Embedding(vocab_size, n_embd)
        
        # Step 3: Positional Encodings
        self. w_pos = nn.Embedding(block_size, n_embd)
        
        # Embedding dropout
        self. embed_dropout = nn.Dropout(dropout)
        
        # Steps 5-11: Stack of attention layers
        self.layers = nn. ModuleList([
            SARANAttentionLayer(n_embd, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Steps 13-14: Output Projection with Bias
        self.w_out = nn. Linear(n_embd, vocab_size, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module. weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass: 
        - Steps 1-4: Embeddings
        - Steps 5-11: Repeated for each layer
        - Steps 12-15: Output
        """
        B, T = idx.shape
        
        # Step 1: Input Tokens (idx)
        
        # Step 2: Token Embeddings
        token_emb = self.w_embed(idx)
        
        # Step 3: Positional Encodings
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.w_pos(positions)
        
        # Step 4: Embedding Summation
        x = token_emb + pos_emb
        x = self.embed_dropout(x)
        
        # Steps 5-11: Apply each attention layer
        for layer in self.layers:
            x = layer(x)
        
        # Step 12: Last Token Selection (all positions for training)
        # Step 13-14: Output Projection with Bias
        logits = self.w_out(x)
        
        # Step 15: Softmax (in loss function or generation)
        
        if targets is None:
            loss = None
        else: 
            logits_flat = logits. view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[: , -1, : ] / temperature
            
            if top_k is not None: 
                v, _ = torch. topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[: , [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch. multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# =============================================================================
# Model Initialization
# =============================================================================
print("=" * 60)
print("SARAN-Deep:  Multi-Layer Attention Network")
print("=" * 60)
print(f"Embedding dimension: {n_embd}")
print(f"Context length: {block_size}")
print(f"Number of layers: {n_layer}")
print(f"Vocabulary size: {vocab_size}")
print("=" * 60)

model = SARANDeep(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_layer=n_layer,
    dropout=dropout
)
model = model.to(device)

n_params = sum(p.numel() for p in model. parameters())
print(f"{n_params / 1e6:. 2f}M parameters")

# =============================================================================
# Optimizer and Scheduler
# =============================================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    max_iters,
    eta_min=learning_rate / 10
)

# =============================================================================
# Training Loop
# =============================================================================
print("\nStarting training...")
print("-" * 60)

best_val_loss = float("inf")

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {it}: train loss {losses['train']:. 4f}, "
            f"val loss {losses['val']:.4f}, "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )
        if losses["val"] < best_val_loss: 
            best_val_loss = losses["val"]
            torch. save(model.state_dict(), "saran_deep_best.pt")
            print(f"  -> New best model saved!  (val_loss:  {best_val_loss:. 4f})")
    
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        (loss / grad_accum_steps).backward()
    
    torch.nn.utils. clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

# =============================================================================
# Save Final Checkpoint
# =============================================================================
checkpoint = {
    "model_state_dict":  model.state_dict(),
    "optimizer_state_dict":  optimizer.state_dict(),
    "config": {
        "block_size": block_size,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "dropout": dropout,
        "vocab_size": vocab_size,
    },
    "iter": max_iters,
    "best_val_loss":  best_val_loss,
}
torch.save(checkpoint, "saran_deep_pretrained.pt")
print("\n" + "=" * 60)
print("Training complete!")
print(f"Best val_loss: {best_val_loss:.4f}")
print("=" * 60)

# =============================================================================
# Test Generation
# =============================================================================
print("\nGenerating sample text...")
print("-" * 60)

model.load_state_dict(torch.load("saran_deep_best.pt", map_location=device))
model.eval()

prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In a world where",
]

for prompt in prompts:
    print(f"\n[Prompt: '{prompt}']")
    prompt_tokens = encode(prompt)
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
    print(decode(generated[0]. tolist()))

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)

tokens_f. close()
