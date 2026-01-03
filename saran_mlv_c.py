"""
SARAN-MLV: Shallow Auto-Regressive Attention Network (Chat Interface)

===============================================================================
PROFESSIONAL-GRADE CONVERSATIONAL AI CHATBOT
===============================================================================

This script provides an interactive chat interface using a fine-tuned SARAN
model. The architecture remains identical to saran_mlv.py and saran_mlv_ft.py.

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
CHAT INTERFACE FEATURES:
===============================================================================
1. Conversation History - maintains context across turns
2. Token Streaming - real-time token-by-token generation
3. Stop Sequences - clean response termination
4. Temperature Control - adjustable creativity
5. Top-K Sampling - focused vocabulary selection

===============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast
import tiktoken

# =============================================================================
# Reproducibility
# =============================================================================
torch.manual_seed(1337)

# =============================================================================
# Hyperparameters (Inference)
# =============================================================================
block_size = 512
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

n_embd = 768
n_layer = 12

# =============================================================================
# Generation Parameters
# =============================================================================
max_new_tokens = 256
temperature = 0.7
top_k = 50
top_p = 0.9

# =============================================================================
# Tokenizer
# =============================================================================
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, disallowed_special=())
decode = lambda l: enc.decode(list(l))


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

        # Dropout for regularization (not used during inference)
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

    @torch.no_grad()
    def generate_streaming(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        stop_tokens=None,
    ):
        """
        Generate text autoregressively with streaming output.

        Args:
            idx: Starting token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            stop_tokens: List of token sequences that terminate generation
        """
        stop_tokens = stop_tokens or []
        generated_tokens = []

        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits, _ = self(idx[:, -self.block_size :])

            # Get last position logits and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Optional nucleus (top-p) sampling
            if top_p:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)

            # Get the generated token
            token_id = next_token[0].item()
            generated_tokens.append(token_id)

            # Yield the token for streaming
            yield token_id

            # Check for stop sequences
            generated_text = decode(generated_tokens)
            for stop in stop_tokens:
                if stop in generated_text:
                    return


# =============================================================================
# Conversation Manager - Handles Multi-Turn Dialogue
# =============================================================================
class ConversationManager:
    """
    Manages conversation history and context for multi-turn dialogue.

    Features:
        - Maintains conversation history within context window
        - Formats prompts for the model
        - Handles context truncation when needed
    """

    def __init__(self, model, max_context=400):
        self.model = model
        self.max_context = max_context
        self.history = []
        self.system_prompt = (
            "You are SARAN, a helpful AI assistant. "
            "You provide clear, accurate, and thoughtful responses.\n\n"
        )

    def add_turn(self, role, content):
        """Add a conversation turn to history."""
        self.history.append({"role": role, "content": content})

    def build_prompt(self, user_input):
        """Build the full prompt including conversation history."""
        prompt = self.system_prompt

        # Add conversation history
        for turn in self.history:
            if turn["role"] == "user":
                prompt += f"User: {turn['content']}\n"
            else:
                prompt += f"Assistant: {turn['content']}\n"

        # Add current user input
        prompt += f"User: {user_input}\nAssistant:"

        # Truncate if too long
        tokens = encode(prompt)
        if len(tokens) > self.max_context:
            # Keep system prompt and truncate history
            excess = len(tokens) - self.max_context
            while excess > 0 and len(self.history) > 0:
                removed = self.history.pop(0)
                excess -= len(encode(f"{removed['role']}: {removed['content']}\n"))
            prompt = self.build_prompt(user_input)

        return prompt

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def generate_response(self, user_input, stream=True):
        """Generate a response to user input."""
        prompt = self.build_prompt(user_input)
        tokens = torch.tensor([encode(prompt)], device=device)

        response_tokens = []
        response_text = ""

        # Stop sequences for clean termination
        stop_sequences = ["\nUser:", "\n\nUser:", "User:", "\n\n\n"]

        if stream:
            # Streaming generation
            for token_id in self.model.generate_streaming(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_sequences,
            ):
                response_tokens.append(token_id)
                chunk = decode([token_id])
                response_text += chunk

                # Check for stop sequences
                should_stop = False
                for stop in stop_sequences:
                    if stop in response_text:
                        response_text = response_text.split(stop)[0]
                        should_stop = True
                        break

                if should_stop:
                    break

                # Print token immediately for streaming effect
                print(chunk, end="", flush=True)

            print()  # Newline after response
        else:
            # Non-streaming generation
            for token_id in self.model.generate_streaming(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_sequences,
            ):
                response_tokens.append(token_id)
                response_text = decode(response_tokens)

                # Check for stop sequences
                for stop in stop_sequences:
                    if stop in response_text:
                        response_text = response_text.split(stop)[0]
                        break

            print(response_text)

        # Clean up response
        response_text = response_text.strip()

        # Add to history
        self.add_turn("user", user_input)
        self.add_turn("assistant", response_text)

        return response_text


# =============================================================================
# Model Initialization
# =============================================================================
print("=" * 70)
print("SARAN-MLV: Conversational AI Chat Interface")
print("=" * 70)
print(f"  Embedding dimension:  {n_embd}")
print(f"  Context length:       {block_size}")
print(f"  Number of layers:     {n_layer}")
print(f"  Vocabulary size:      {vocab_size}")
print(f"  FFN expansion ratio:  2x (vs 4x in GPT)")
print(f"  Attention heads:      1 (single-head, vs 12 in GPT)")
print("=" * 70)

model = SARANMLV(vocab_size, n_embd, block_size, n_layer, dropout=0.0)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params / 1e6:.2f}M")

# Compile model for faster execution (PyTorch 2.0+)
if hasattr(torch, "compile") and device == "cuda":
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")

# =============================================================================
# Load Fine-Tuned Weights
# =============================================================================
model_paths = [
    "saran_mlv_finetuned.pt",
    "saran_mlv_ft_best.pt",
    "saran_mlv_best.pt",
    "saran_mlv_pretrained.pt",
]

loaded = False
for path in model_paths:
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {path}")
            if "best_val_loss" in checkpoint:
                print(f"  Model val loss: {checkpoint['best_val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        loaded = True
        break

if not loaded:
    print("ERROR: No model weights found!")
    print(
        "Please run saran_mlv.py (pre-training) and saran_mlv_ft.py (fine-tuning) first."
    )
    sys.exit(1)

model.eval()
print("=" * 70)

# =============================================================================
# Initialize Conversation Manager
# =============================================================================
conversation = ConversationManager(model)

# =============================================================================
# Chat Interface
# =============================================================================
print("\n" + "=" * 70)
print("SARAN Chat - Type 'quit' to exit, 'clear' to reset conversation")
print("=" * 70 + "\n")


def print_help():
    """Print help information."""
    print("\n" + "-" * 50)
    print("Commands:")
    print("  quit, exit, q  - Exit the chat")
    print("  clear, reset   - Clear conversation history")
    print("  help, ?        - Show this help message")
    print("  temp <value>   - Set temperature (0.1-2.0)")
    print("  topk <value>   - Set top-k (1-100)")
    print("  topp <value>   - Set top-p (0.1-1.0)")
    print("-" * 50 + "\n")


# Main chat loop
while True:
    try:
        user_input = input("\033[94mYou:\033[0m ").strip()

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! Thank you for chatting with SARAN.")
            break

        elif user_input.lower() in ["clear", "reset"]:
            conversation.clear_history()
            print("\n[Conversation history cleared]\n")
            continue

        elif user_input.lower() in ["help", "?"]:
            print_help()
            continue

        elif user_input.lower().startswith("temp "):
            try:
                new_temp = float(user_input.split()[1])
                if 0.1 <= new_temp <= 2.0:
                    temperature = new_temp
                    print(f"\n[Temperature set to {temperature}]\n")
                else:
                    print("\n[Temperature must be between 0.1 and 2.0]\n")
            except ValueError:
                print("\n[Invalid temperature value]\n")
            continue

        elif user_input.lower().startswith("topk "):
            try:
                new_topk = int(user_input.split()[1])
                if 1 <= new_topk <= 100:
                    top_k = new_topk
                    print(f"\n[Top-K set to {top_k}]\n")
                else:
                    print("\n[Top-K must be between 1 and 100]\n")
            except ValueError:
                print("\n[Invalid top-k value]\n")
            continue

        elif user_input.lower().startswith("topp "):
            try:
                new_topp = float(user_input.split()[1])
                if 0.1 <= new_topp <= 1.0:
                    top_p = new_topp
                    print(f"\n[Top-P set to {top_p}]\n")
                else:
                    print("\n[Top-P must be between 0.1 and 1.0]\n")
            except ValueError:
                print("\n[Invalid top-p value]\n")
            continue

        # Generate response
        print("\033[92mSARAN:\033[0m ", end="", flush=True)
        conversation.generate_response(user_input, stream=True)
        print()

    except KeyboardInterrupt:
        print("\n\nGoodbye! Thank you for chatting with SARAN.")
        break
    except Exception as e:
        print(f"\n[Error: {e}]\n")
        continue

print("\n" + "=" * 70)
print("Chat session ended")
print("=" * 70)
