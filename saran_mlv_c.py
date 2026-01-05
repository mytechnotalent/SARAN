"""
SARAN-MLV Chat Interface

===============================================================================
CHAT INTERFACE FOR SARAN-MLV
===============================================================================

This module provides an interactive chat interface for the SARAN-MLV model.
It includes:

1. Web Search Integration
   - Automatic web search for questions (ending with ?)
   - Configurable search triggers in config.json
   - DuckDuckGo search via web.py agent

2. Garbage Detection
   - Detects low-quality model outputs
   - Responds with "I don't know" for garbage outputs
   - Checks for repetition, special characters, short responses

3. Conversation History
   - Maintains last 10 conversation turns
   - Supports clear/reset commands
   - Stop sequences to prevent runaway generation

===============================================================================
"""

import json
import os

import tiktoken
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn import functional as F

import web

# =============================================================================
# Configuration
# =============================================================================
cfg = json.load(open("config.json")) if os.path.exists("config.json") else {}
mcfg = cfg.get("model", {})
gcfg = cfg.get("generation", {})

# Model hyperparameters
B = mcfg.get("block_size", 512)  # Context length
D = mcfg.get("n_embd", 768)  # Embedding dimension
L = mcfg.get("n_layer", 12)  # Number of transformer layers
V = mcfg.get("vocab_size", 50304)  # Vocabulary size (aligned for GPU efficiency)

# Device selection
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")


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
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# =============================================================================
# Attention Layer - Single Head (SARAN Innovation)
# =============================================================================
class Attn(nn.Module):
    """
    SARAN's Single-Head Attention Layer.

    Unlike GPT-2 which uses 12 attention heads, SARAN uses a SINGLE head.
    This is simpler, more interpretable, and equally effective.

    Uses Flash Attention via F.scaled_dot_product_attention for efficiency.
    """

    def __init__(self, dim):
        super().__init__()
        # Fused Q, K, V projection (more efficient than separate projections)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        # Output projection after attention
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Split into Q, K, V
        q, k, v = self.qkv(x).split(x.size(-1), -1)
        # Flash Attention with causal masking
        return self.out_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True))


# =============================================================================
# Feed-Forward Network - 2x Expansion (SARAN Innovation)
# =============================================================================
class FFN(nn.Module):
    """
    SARAN's Feed-Forward Network with 2x expansion.

    GPT-2 uses 4x expansion (768 -> 3072 -> 768).
    SARAN uses 2x expansion (768 -> 1536 -> 768).

    This is more parameter-efficient while maintaining quality.
    Uses SiLU (Swish) activation instead of GELU.
    """

    def __init__(self, dim):
        super().__init__()
        hidden = dim * 2  # 2x expansion (SARAN innovation, vs 4x in GPT)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


# =============================================================================
# Transformer Block - Attention + FFN + Residuals
# =============================================================================
class Block(nn.Module):
    """
    One SARAN transformer block.

    Architecture (Pre-Norm style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, dim):
        super().__init__()
        # Pre-normalization layers
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        # Attention and FFN
        self.attn = Attn(dim)
        self.ffn = FFN(dim)

    def forward(self, x):
        # Residual connections around attention and FFN
        return x + self.ffn(self.ln2(x + self.attn(self.ln1(x))))


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
    """

    def __init__(self):
        super().__init__()
        # Token and position embeddings
        self.wte = nn.Embedding(V, D)
        self.wpe = nn.Embedding(B, D)
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(D) for _ in range(L)])
        # Final normalization and output projection
        self.ln_f = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        # Weight tying: share embedding and output weights
        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        """
        Forward pass through the model.

        Args:
            idx: Input token indices (batch, seq_len)

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        # Combine token and positional embeddings
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        # Pass through transformer blocks
        for b in self.blocks:
            x = b(x)
        # Final norm and output projection
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_tok, temp=0.7, top_k=40, rep=1.3):
        """
        Generate text autoregressively with streaming.

        Args:
            idx: Starting token indices (batch, seq_len)
            max_tok: Maximum number of tokens to generate
            temp: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens
            rep: Repetition penalty (> 1.0 discourages repetition)

        Yields:
            token: Each generated token ID
        """
        for _ in range(max_tok):
            # Forward pass with mixed precision
            with autocast(
                device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"
            ):
                logits = self(idx[:, -B:])[:, -1, :]

            # Apply repetition penalty
            for t in set(idx[0].tolist()):
                logits[0, t] /= rep if logits[0, t] > 0 else 1 / rep

            # Apply temperature
            logits /= temp

            # Top-k filtering
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

            # Sample next token
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
            yield idx[0, -1].item()


# =============================================================================
# Garbage Detection
# =============================================================================
def is_garbage(text):
    """
    Detect garbage/low-quality model output.

    Checks for:
        - Empty or very short responses
        - Too few words
        - Excessive special characters
        - Repetitive words (low unique word ratio)

    Args:
        text: Generated text to evaluate

    Returns:
        bool: True if output appears to be garbage
    """
    # Check for empty or very short text
    if not text or len(text) < 10:
        return True

    # Check for too few words
    words = text.split()
    if len(words) < 3:
        return True

    # Check for excessive special characters (>30% of text)
    special = sum(1 for c in text if c in "�-,./()[]{}|\\;:'\"!@#$%^&*")
    if special > len(text) * 0.3:
        return True

    # Check for repetitive words (<30% unique)
    if len(set(words)) < len(words) * 0.3:
        return True

    return False


# =============================================================================
# Chat Interface
# =============================================================================
def chat(model):
    """
    Interactive chat loop with the SARAN model.

    Features:
        - Web search for questions (ending with ?)
        - Conversation history (last 10 turns)
        - Garbage detection with fallback response
        - Commands: quit/exit/q, clear/reset

    Args:
        model: Loaded SARAN model instance
    """
    # Load generation parameters from config
    temp = gcfg.get("temperature", 0.7)
    top_k = gcfg.get("top_k", 40)
    rep = gcfg.get("repetition_penalty", 1.3)
    max_tok = gcfg.get("max_new_tokens", 256)
    triggers = set(cfg.get("search_triggers", []))

    # Conversation state
    history = []
    stops = ["\nUser:", "User:", "\n\n\n"]

    print(f"\nSARAN | temp={temp} top_k={top_k} rep={rep}\n")

    while True:
        try:
            # Get user input
            q = input("\033[94mYou:\033[0m ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.lower() in ("clear", "reset"):
                history.clear()
                print("[cleared]\n")
                continue

            # Web search for questions
            if q.endswith("?") or (q.split() and q.split()[0].lower() in triggers):
                r = web.search(q)
                if r:
                    print(f"\n\033[92mSARAN:\033[0m {r}\n")
                    history.append({"u": q, "a": r})
                    continue
                print("\033[90m[not found]\033[0m")
                print("\033[92mSARAN:\033[0m I don't know.\n")
                continue

            # Build prompt with conversation history
            prompt = "".join(
                f"User: {h['u']}\nAssistant: {h['a']}\n" for h in history[-10:]
            )
            prompt += f"User: {q}\nAssistant:"

            # Encode and generate
            idx = torch.tensor(
                [enc.encode(prompt, disallowed_special=())], device=device
            )

            # Stream generation with stop sequence detection
            resp = ""
            for tok in model.generate(idx, max_tok, temp, top_k, rep):
                resp += enc.decode([tok], errors="replace")
                for s in stops:
                    if s in resp:
                        resp = resp.split(s)[0]
                        break
                else:
                    continue
                break

            # Output response or fallback
            resp = resp.strip()
            if is_garbage(resp):
                print("\033[92mSARAN:\033[0m I don't know.\n")
            else:
                print(f"\033[92mSARAN:\033[0m {resp}\n")
                history.append({"u": q, "a": resp})

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    print(f"Device: {device}")

    # Initialize model
    model = SARAN().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load fine-tuned weights
    ckpt = torch.load("saran_mlv_ft_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print("Loaded")

    # Start chat
    model.eval()
    chat(model)
