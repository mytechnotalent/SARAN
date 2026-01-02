"""
SARAN:  Shallow Auto-Regressive Attention Network (PyTorch Implementation)

A single-layer, single-head attention model following the original 15-step architecture: 
1. Input Tokens
2. Token Embeddings
3. Positional Encodings
4. Embedding Summation
5. Query Projection
6. Key Projection
7. Value Projection
8. Attention Score Calculation
9. Causal Masking
10. Softmax
11. Attention Output Calculation
12. Last Token Selection
13. Output Projection
14. Bias Addition
15. Softmax Activation

Trained on OpenWebText dataset.
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
grad_accum_steps = 8  # effective batch = 8 * 8 = 64
block_size = 256      # context length (SARAN original: 4, scaled up)
max_iters = 20000
eval_interval = 500
learning_rate = 3e-4
device = (
    "mps"
    if torch. backends.mps.is_available()
    else "cuda" if torch.cuda. is_available() else "cpu"
)
print(f"Using device: {device}")
eval_iters = 200
n_embd = 640          # embedding dimension (SARAN original:  32, scaled up)
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
        line = tokens_f. readline()
        toks = json.loads(line. decode("utf-8"))
        L = len(toks)
        if L <= block_size: 
            continue  # skip short documents
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
# SARAN Model (15-Step Architecture)
# =============================================================================
class SARAN(nn. Module):
    """
    Shallow Auto-Regressive Attention Network
    
    Single-layer, single-head causal self-attention following the original
    15-step forward pass architecture.  This is intentionally shallow to
    demonstrate that meaningful language modeling can be achieved without
    deep transformer stacks.
    
    Architecture differences from GPT:
    - Single attention layer (no stacking)
    - Single attention head (no multi-head)
    - No feed-forward network after attention
    - No layer normalization
    - Direct output projection from attention output
    """
    
    def __init__(self, vocab_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.scale = n_embd ** 0.5
        
        # Step 2: Token Embeddings (vocab_size x n_embd)
        self.w_embed = nn.Embedding(vocab_size, n_embd)
        
        # Step 3: Positional Encodings (block_size x n_embd)
        self.w_pos = nn. Embedding(block_size, n_embd)
        
        # Steps 5, 6, 7: Query, Key, Value Projections (n_embd x n_embd each)
        self.w_q = nn.Linear(n_embd, n_embd, bias=False)
        self.w_k = nn.Linear(n_embd, n_embd, bias=False)
        self.w_v = nn. Linear(n_embd, n_embd, bias=False)
        
        # Steps 13, 14: Output Projection with Bias (n_embd -> vocab_size)
        self.w_out = nn.Linear(n_embd, vocab_size, bias=True)
        
        # Dropout for regularization
        self.embed_dropout = nn. Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Step 9: Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )
        
        # Initialize weights (similar to original SARAN)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values like original implementation."""
        nn.init.normal_(self.w_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_q. weight, mean=0.0, std=0.02)
        nn.init. normal_(self.w_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)
        nn.init.normal_(self. w_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_out.bias)
    
    def forward(self, idx, targets=None):
        """
        Forward pass following the 15-step SARAN architecture. 
        
        Unlike the original which only predicted the next token after the last
        position, this version predicts next tokens at ALL positions for
        efficient training (standard language modeling approach).
        
        Args:
            idx: Input token IDs of shape (B, T)
            targets: Target token IDs of shape (B, T) for training
            
        Returns: 
            logits:  Output logits of shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        
        # Step 1: Input Tokens (already provided as idx)
        
        # Step 2: Token Embeddings
        token_emb = self.w_embed(idx)  # (B, T, n_embd)
        
        # Step 3: Positional Encodings
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.w_pos(positions)  # (T, n_embd)
        
        # Step 4:  Embedding Summation
        x = token_emb + pos_emb  # (B, T, n_embd)
        x = self.embed_dropout(x)
        
        # Step 5: Query Projection
        q = self.w_q(x)  # (B, T, n_embd)
        
        # Step 6: Key Projection
        k = self. w_k(x)  # (B, T, n_embd)
        
        # Step 7: Value Projection
        v = self.w_v(x)  # (B, T, n_embd)
        
        # Step 8: Attention Score Calculation
        # scores = (Q @ K^T) / sqrt(d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, T, T)
        
        # Step 9: Causal Masking
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        
        # Step 10: Softmax
        attn = F.softmax(scores, dim=-1)  # (B, T, T)
        attn = self.attn_dropout(attn)
        
        # Step 11: Attention Output Calculation
        attn_out = torch.matmul(attn, v)  # (B, T, n_embd)
        
        # Step 12: Last Token Selection
        # NOTE: For training efficiency, we predict at ALL positions, not just last
        # The original SARAN only used last = attn_out[:, -1, : ] for single prediction
        # Here we keep all positions for parallel training
        
        # Steps 13 & 14: Output Projection with Bias
        logits = self.w_out(attn_out)  # (B, T, vocab_size)
        
        # Step 15: Softmax Activation
        # Applied in loss function (cross_entropy) or during generation
        
        # Compute loss if targets provided
        if targets is None:
            loss = None
        else: 
            # Reshape for cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively. 
        
        Args:
            idx:  Starting token IDs of shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens
            
        Returns:
            idx: Generated token IDs of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position only
            logits = logits[:, -1, : ] / temperature  # (B, vocab_size)
            
            # Optional top-k filtering
            if top_k is not None: 
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[: , [-1]]] = float('-inf')
            
            # Step 15: Softmax Activation (explicit during generation)
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch. multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch. cat((idx, idx_next), dim=1)
        
        return idx


# =============================================================================
# Model Initialization
# =============================================================================
print("=" * 60)
print("SARAN:  Shallow Auto-Regressive Attention Network")
print("=" * 60)
print(f"Embedding dimension: {n_embd}")
print(f"Context length: {block_size}")
print(f"Vocabulary size: {vocab_size}")
print("Architecture:  Single-layer, Single-head Attention")
print("=" * 60)

model = SARAN(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout
)
model = model.to(device)

n_params = sum(p.numel() for p in model. parameters())
print(f"{n_params / 1e6:.2f}M parameters")

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
    # Evaluation
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {it}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )
        # Save best model
        if losses["val"] < best_val_loss: 
            best_val_loss = losses["val"]
            torch.save(model. state_dict(), "saran_best. pt")
            print(f"  -> New best model saved!  (val_loss:  {best_val_loss:.4f})")
    
    # Gradient accumulation training step
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        (loss / grad_accum_steps).backward()
    
    # Gradient clipping
    torch.nn. utils.clip_grad_norm_(model. parameters(), grad_clip)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()

# =============================================================================
# Save Final Checkpoint
# =============================================================================
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": {
        "block_size":  block_size,
        "n_embd": n_embd,
        "dropout": dropout,
        "vocab_size":  vocab_size,
    },
    "iter": max_iters,
    "best_val_loss": best_val_loss,
}
torch.save(checkpoint, "saran_pretrained.pt")
print("\n" + "=" * 60)
print("Training complete!")
print(f"Final model saved to saran_pretrained.pt")
print(f"Best model saved to saran_best.pt (val_loss: {best_val_loss:.4f})")
print("=" * 60)

# =============================================================================
# Test Generation
# =============================================================================
print("\nGenerating sample text...")
print("-" * 60)

# Load best model for generation
model.load_state_dict(torch.load("saran_best.pt", map_location=device))
model.eval()

# Generate from empty context
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model. generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
print("\n[Generated from empty context]:")
print(decode(generated[0]. tolist()))

# Generate from prompt
print("\n" + "-" * 60)
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

# Cleanup
tokens_f.close()
