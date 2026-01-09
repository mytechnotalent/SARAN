"""
SARAN-MLV Fine-Tuning Script

===============================================================================
FINE-TUNING SARAN ON INSTRUCTION DATA (SFT + DPO)
===============================================================================

This script fine-tunes the pretrained SARAN-MLV model in two phases:

Phase 1 - Supervised Fine-Tuning (SFT):
    1. Load pretrained weights from saran_mlv_best.pt
    2. Train on Alpaca instruction-response pairs
    3. Early stopping based on validation loss
    4. Save checkpoint to saran_mlv_ft_best.pt

Phase 2 - Direct Preference Optimization (DPO):
    1. Load SFT model as policy and frozen reference
    2. Train on human preference data (chosen vs rejected)
    3. DPO loss optimizes policy to prefer chosen responses
    4. Save final model to saran_mlv_dpo_best.pt

Dataset Formats:
    SFT:  User: <instruction>\nAssistant: <response>
    DPO:  (prompt, chosen_response, rejected_response) triples

Key Hyperparameters:
    SFT: lr=3e-5, dropout=0.1, early stopping
    DPO: lr=1e-6, beta=0.1 (KL penalty), reference model frozen

===============================================================================
"""

import copy
import json
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
# Configuration (loaded from config.json)
# =============================================================================
cfg = json.load(open("config.json")) if os.path.exists("config.json") else {}
mcfg = cfg.get("model", {})
fcfg = cfg.get("finetuning", {})
dcfg = cfg.get("dpo", {})

# Model hyperparameters
block_size = mcfg.get("block_size", 512)
n_embd = mcfg.get("n_embd", 1536)
n_layer = mcfg.get("n_layer", 24)
vocab_size = mcfg.get("vocab_size", 50304)
dropout = mcfg.get("dropout", 0.1)

# SFT hyperparameters
sft_batch_size = fcfg.get("batch_size", 2)
sft_grad_accum_steps = fcfg.get("grad_accum_steps", 8)
sft_max_iters = fcfg.get("max_iters", 50000)
sft_eval_interval = fcfg.get("eval_interval", 200)
sft_learning_rate = fcfg.get("learning_rate", 3e-5)
sft_grad_clip = fcfg.get("grad_clip", 1.0)
sft_weight_decay = fcfg.get("weight_decay", 0.01)
sft_patience = fcfg.get("patience", 5)

# DPO hyperparameters
dpo_enabled = dcfg.get("enabled", True)
dpo_beta = dcfg.get("beta", 0.1)
dpo_learning_rate = dcfg.get("learning_rate", 1e-6)
dpo_max_iters = dcfg.get("max_iters", 5000)
dpo_eval_interval = dcfg.get("eval_interval", 100)
dpo_batch_size = dcfg.get("batch_size", 1)
dpo_grad_accum_steps = dcfg.get("grad_accum_steps", 16)
dpo_grad_clip = dcfg.get("grad_clip", 1.0)
dpo_weight_decay = dcfg.get("weight_decay", 0.01)
dpo_patience = dcfg.get("patience", 10)
dpo_dataset = dcfg.get("dataset", "Anthropic/hh-rlhf")

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


def get_batch(split_name, batch_sz=None):
    """
    Get a batch of data for training or validation.

    Args:
        split_name: "train" or "val"
        batch_sz: Optional batch size override

    Returns:
        x: Input tokens (batch_size, block_size)
        y: Target tokens (batch_size, block_size)
    """
    d = train_data if split_name == "train" else val_data
    bs = batch_sz if batch_sz else sft_batch_size
    ix = torch.randint(len(d) - block_size, (bs,))
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
# Feed-Forward Network - 4x Expansion with Dropout
# =============================================================================
class FFN(nn.Module):
    """
    SARAN's Feed-Forward Network with 4x expansion and dropout.

    Uses standard GPT-style 4x expansion (1536 -> 6144 -> 1536).
    This provides more capacity for knowledge storage and synthesis.
    """

    def __init__(self, dim, drop=0.0):
        """
        Initialize the feed-forward network with dropout.

        Args:
            dim (int): Embedding dimension. The hidden layer will be 4x this size.
            drop (float, optional): Dropout probability applied after the output
                projection. Defaults to 0.0.
        """
        super().__init__()
        hidden = dim * 4
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

# Load pretrained weights (best model from pretraining)
path = "saran_mlv_best.pt"
if os.path.exists(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # saran_mlv_best.pt is just state_dict, not a full checkpoint
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print(f"Loaded: {path}")
else:
    print("WARNING: No pretrained weights found!")

# Optimizer for SFT
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=sft_learning_rate,
    betas=(0.9, 0.95),
    weight_decay=sft_weight_decay,
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
# PHASE 1: Supervised Fine-Tuning (SFT)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Supervised Fine-Tuning (SFT)")
print("=" * 60)
best_val_loss = float("inf")
wait = 0

for it in range(sft_max_iters):
    # Periodic evaluation
    if it % sft_eval_interval == 0:
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
            if wait >= sft_patience:
                print(f"\nEarly stop at {it}")
                break

    # Training step with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    for _ in range(sft_grad_accum_steps):
        xb, yb = get_batch("train")
        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(xb, yb)
        (loss / sft_grad_accum_steps).backward()

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), sft_grad_clip)
    optimizer.step()

print(f"\nSFT Complete! Best val loss: {best_val_loss:.4f}")


# =============================================================================
# PHASE 2: Direct Preference Optimization (DPO)
# =============================================================================
if dpo_enabled:
    print("\n" + "=" * 60)
    print("PHASE 2: Direct Preference Optimization (DPO)")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # DPO Dataset Loading
    # -------------------------------------------------------------------------
    print(f"Loading preference data from: {dpo_dataset}")

    # Try to load from HuggingFace datasets
    try:
        from datasets import load_dataset

        if dpo_dataset == "Anthropic/hh-rlhf":
            # Anthropic's helpful/harmless dataset
            ds = load_dataset("Anthropic/hh-rlhf", split="train[:10000]")
            preference_data = []
            for ex in ds:
                # Extract chosen and rejected from the conversations
                chosen = ex.get("chosen", "")
                rejected = ex.get("rejected", "")
                if chosen and rejected:
                    preference_data.append({"chosen": chosen, "rejected": rejected})
        else:
            # Generic preference dataset format
            ds = load_dataset(dpo_dataset, split="train[:10000]")
            preference_data = []
            for ex in ds:
                chosen = ex.get("chosen", ex.get("preferred", ""))
                rejected = ex.get("rejected", ex.get("dispreferred", ""))
                if chosen and rejected:
                    preference_data.append({"chosen": chosen, "rejected": rejected})

        print(f"Loaded {len(preference_data)} preference pairs")

    except Exception as e:
        print(f"Could not load {dpo_dataset}: {e}")
        print("Creating synthetic preference data from Alpaca...")

        # Fallback: Create synthetic preferences from Alpaca data
        # Use model to generate alternative responses, prefer original
        alpaca = json.load(open("alpaca_data.json"))
        preference_data = []

        for item in alpaca[:5000]:  # Limit to 5k examples
            instr = item["instruction"].strip()
            inp = item.get("input", "").strip()
            out = item["output"].strip()
            user = f"{instr}\n{inp}" if inp else instr

            # Chosen: Original Alpaca response
            chosen = f"User: {user}\nAssistant: {out}"

            # Rejected: Truncated or modified response (simple heuristic)
            # In production, you'd use a reward model or human annotations
            rejected_out = out[: len(out) // 2] + "..."  # Truncated = lower quality
            rejected = f"User: {user}\nAssistant: {rejected_out}"

            preference_data.append({"chosen": chosen, "rejected": rejected})

        print(f"Created {len(preference_data)} synthetic preference pairs")

    # Split into train/val
    dpo_split = int(0.9 * len(preference_data))
    dpo_train = preference_data[:dpo_split]
    dpo_val = preference_data[dpo_split:]
    print(f"DPO Train: {len(dpo_train)} | Val: {len(dpo_val)} pairs")

    # -------------------------------------------------------------------------
    # Reference Model (frozen copy of SFT model)
    # -------------------------------------------------------------------------
    print("Creating reference model (frozen)...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # DPO Loss Function
    # -------------------------------------------------------------------------
    def get_log_probs(model_to_use, tokens):
        """
        Compute log probabilities of tokens under a model.

        Args:
            model_to_use: The model to compute log probs with
            tokens: Input token tensor (batch, seq_len)

        Returns:
            Log probabilities summed over sequence (batch,)
        """
        with torch.no_grad() if not model_to_use.training else torch.enable_grad():
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model_to_use(tokens[:, :-1])
                log_probs = F.log_softmax(logits, dim=-1)

                # Gather log probs of actual next tokens
                targets = tokens[:, 1:]
                gathered = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(
                    -1
                )

                # Sum over sequence (average would also work)
                return gathered.sum(dim=-1)

    def dpo_loss(model_policy, model_ref, chosen_tokens, rejected_tokens, beta):
        """
        Compute DPO loss.

        DPO Loss = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x)
                                        - log_ref(y_w|x) + log_ref(y_l|x))))

        Where:
            - y_w = chosen (winning) response
            - y_l = rejected (losing) response
            - pi = policy model (being trained)
            - ref = reference model (frozen)
            - beta = temperature parameter

        Args:
            model_policy: Policy model being trained
            model_ref: Frozen reference model
            chosen_tokens: Tokenized chosen responses
            rejected_tokens: Tokenized rejected responses
            beta: DPO temperature (higher = more conservative)

        Returns:
            Scalar loss value
        """
        # Get log probs from policy model
        pi_chosen = get_log_probs(model_policy, chosen_tokens)
        pi_rejected = get_log_probs(model_policy, rejected_tokens)

        # Get log probs from reference model (no grad)
        with torch.no_grad():
            ref_chosen = get_log_probs(model_ref, chosen_tokens)
            ref_rejected = get_log_probs(model_ref, rejected_tokens)

        # DPO objective
        log_ratio = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
        loss = -F.logsigmoid(beta * log_ratio).mean()

        return loss, {
            "chosen_reward": (pi_chosen - ref_chosen).mean().item(),
            "rejected_reward": (pi_rejected - ref_rejected).mean().item(),
        }

    # -------------------------------------------------------------------------
    # DPO Data Batching
    # -------------------------------------------------------------------------
    def get_dpo_batch(split_name):
        """
        Get a batch of preference pairs for DPO training.

        Args:
            split_name: "train" or "val"

        Returns:
            chosen_tokens: Tokenized chosen responses (batch, seq_len)
            rejected_tokens: Tokenized rejected responses (batch, seq_len)
        """
        data = dpo_train if split_name == "train" else dpo_val
        batch_indices = torch.randint(len(data), (dpo_batch_size,))

        chosen_list = []
        rejected_list = []

        for idx in batch_indices:
            pair = data[idx.item()]

            # Tokenize and truncate to block_size
            chosen_enc = encode(pair["chosen"])[:block_size]
            rejected_enc = encode(pair["rejected"])[:block_size]

            # Pad to block_size
            chosen_pad = chosen_enc + [0] * (block_size - len(chosen_enc))
            rejected_pad = rejected_enc + [0] * (block_size - len(rejected_enc))

            chosen_list.append(chosen_pad)
            rejected_list.append(rejected_pad)

        chosen_tokens = torch.tensor(chosen_list, dtype=torch.long, device=device)
        rejected_tokens = torch.tensor(rejected_list, dtype=torch.long, device=device)

        return chosen_tokens, rejected_tokens

    # -------------------------------------------------------------------------
    # DPO Loss Estimation
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def estimate_dpo_loss():
        """
        Estimate DPO loss on train and validation sets.

        Returns:
            dict: {"train": avg_train_loss, "val": avg_val_loss}
        """
        model.eval()
        out = {}
        for split_name in ["train", "val"]:
            losses = []
            for _ in range(25):  # Fewer eval steps for DPO
                chosen, rejected = get_dpo_batch(split_name)
                loss, _ = dpo_loss(model, ref_model, chosen, rejected, dpo_beta)
                losses.append(loss.item())
            out[split_name] = sum(losses) / len(losses)
        model.train()
        return out

    # -------------------------------------------------------------------------
    # DPO Training Loop
    # -------------------------------------------------------------------------
    # New optimizer for DPO (lower learning rate)
    dpo_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dpo_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=dpo_weight_decay,
    )

    best_dpo_loss = float("inf")
    dpo_wait = 0

    print(f"\nStarting DPO training (beta={dpo_beta}, lr={dpo_learning_rate})...")

    for it in range(dpo_max_iters):
        # Periodic evaluation
        if it % dpo_eval_interval == 0:
            losses = estimate_dpo_loss()
            print(
                f"DPO step {it:>5}: train {losses['train']:.4f}, val {losses['val']:.4f}"
            )

            # Save best model and check early stopping
            if losses["val"] < best_dpo_loss:
                best_dpo_loss = losses["val"]
                dpo_wait = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_dpo_loss": best_dpo_loss,
                        "best_sft_loss": best_val_loss,
                    },
                    "saran_mlv_dpo_best.pt",
                )
                print(f"         -> saved (dpo_val={best_dpo_loss:.4f})")
            else:
                dpo_wait += 1
                if dpo_wait >= dpo_patience:
                    print(f"\nDPO Early stop at {it}")
                    break

        # Training step with gradient accumulation
        dpo_optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(dpo_grad_accum_steps):
            chosen, rejected = get_dpo_batch("train")
            loss, metrics = dpo_loss(model, ref_model, chosen, rejected, dpo_beta)
            (loss / dpo_grad_accum_steps).backward()
            accum_loss += loss.item() / dpo_grad_accum_steps

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), dpo_grad_clip)
        dpo_optimizer.step()

    print(f"\nDPO Complete! Best DPO val loss: {best_dpo_loss:.4f}")
    print(f"Final model saved to: saran_mlv_dpo_best.pt")

else:
    print("\nDPO disabled. Using SFT model as final output.")
    print(f"Final model saved to: saran_mlv_ft_best.pt")

print("\n" + "=" * 60)
print("Fine-tuning Pipeline Complete!")
print("=" * 60)
