"""
SARAN-MLV Fine-Tuning on Alpaca
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
torch.manual_seed(1337)

# =============================================================================
# Config
# =============================================================================
batch_size = 4
grad_accum_steps = 4
block_size = 512
max_iters = 5000
eval_interval = 200
learning_rate = 3e-5
dropout = 0.1
grad_clip = 1.0
patience = 5

n_embd = 768
n_layer = 12
vocab_size = 50304

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
# Dataset
# =============================================================================
finetune_file = "finetune_alpaca.txt"

if not os.path.exists(finetune_file):
    print("Downloading Alpaca dataset...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "alpaca_data.json",
    )
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

data = torch.tensor(encode(open(finetune_file).read()), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"Train: {len(train_data):,} | Val: {len(val_data):,} tokens")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# =============================================================================
# Model
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SARANAttentionLayer(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x):
        q, k, v = self.qkv(x).split(x.size(-1), dim=-1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out)


class SARANFFN(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_embd * 2, bias=False)
        self.w2 = nn.Linear(n_embd * 2, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))))


class SARANBlock(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.attn = SARANAttentionLayer(n_embd, dropout)
        self.ffn = SARANFFN(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SARANMLV(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [SARANBlock(n_embd, dropout) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        )
        return logits, loss


# =============================================================================
# Training
# =============================================================================
model = SARANMLV(vocab_size, n_embd, block_size, n_layer, dropout).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Load pretrained
for path in ["saran_mlv_pretrained.pt", "saran_mlv_best.pt"]:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Loaded: {path}")
        break
else:
    print("WARNING: No pretrained weights!")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = [0.0] * 50
        for k in range(50):
            X, Y = get_batch(split)
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


print("\nFine-tuning...")
best_val_loss = float("inf")
wait = 0

for it in range(max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:>5}: train {losses['train']:.4f}, val {losses['val']:.4f}")
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

    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        xb, yb = get_batch("train")
        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(xb, yb)
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

print(f"\nDone! Best val loss: {best_val_loss:.4f}")
