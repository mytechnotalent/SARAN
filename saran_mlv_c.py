"""
SARAN-MLV Chat Interface
Minimal, clean implementation for conversational inference.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast
import tiktoken

# =============================================================================
# Config
# =============================================================================
torch.manual_seed(1337)

block_size = 512
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

# Generation params (adjustable via commands)
max_new_tokens = 256
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.3

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, disallowed_special=())
decode = lambda l: enc.decode(l, errors="replace")  # Fix weird chars


# =============================================================================
# Model Components
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SARANAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).split(x.size(-1), dim=-1)
        return self.out_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True))


class SARANFFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_embd * 2, bias=False)
        self.w2 = nn.Linear(n_embd * 2, n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class SARANBlock(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.attn = SARANAttention(n_embd)
        self.ffn = SARANFFN(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SARANMLV(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer):
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

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_tokens, temp, top_k, top_p, rep_penalty):
        for _ in range(max_tokens):
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = self(idx[:, -self.block_size :])[:, -1, :]

            # Repetition penalty
            if rep_penalty != 1.0:
                for tid in set(idx[0].tolist()):
                    logits[0, tid] /= (
                        rep_penalty if logits[0, tid] > 0 else (1 / rep_penalty)
                    )

            logits /= temp

            # Top-k
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)
            yield next_tok[0].item()


# =============================================================================
# Chat
# =============================================================================
def clean_response(text):
    """Clean up response text."""
    text = text.strip()
    # Remove trailing incomplete sentence
    if text and text[-1] not in ".!?":
        for p in ".!?":
            i = text.rfind(p)
            if i > len(text) // 2:  # Only if we have at least half a response
                text = text[: i + 1]
                break
    return text


def chat_loop(model):
    global temperature, top_k, top_p, repetition_penalty

    history = []
    stop_seqs = ["\nUser:", "User:", "\n\n\n"]

    print(f"\nSARAN Chat | temp={temperature} top_k={top_k} rep={repetition_penalty}")
    print("Commands: quit, clear, temp/topk/topp/rep <val>, help\n")

    while True:
        try:
            user_in = input("\033[94mYou:\033[0m ").strip()
            if not user_in:
                continue

            # Commands
            cmd = user_in.lower()
            if cmd in ("quit", "exit", "q"):
                break
            if cmd in ("clear", "reset"):
                history.clear()
                print("[Cleared]\n")
                continue
            if cmd in ("help", "?"):
                print(
                    "Commands: quit, clear, temp <0.1-2>, topk <1-100>, topp <0.1-1>, rep <1-2>\n"
                )
                continue

            # Parameter commands
            if cmd.startswith("temp "):
                temperature = max(0.1, min(2.0, float(cmd.split()[1])))
                print(f"[temp={temperature}]\n")
                continue
            if cmd.startswith("topk "):
                top_k = max(1, min(100, int(cmd.split()[1])))
                print(f"[top_k={top_k}]\n")
                continue
            if cmd.startswith("topp "):
                top_p = max(0.1, min(1.0, float(cmd.split()[1])))
                print(f"[top_p={top_p}]\n")
                continue
            if cmd.startswith("rep "):
                repetition_penalty = max(1.0, min(2.0, float(cmd.split()[1])))
                print(f"[rep={repetition_penalty}]\n")
                continue

            # Build prompt
            prompt = "".join(f"User: {h['u']}\nAssistant: {h['a']}\n" for h in history)
            prompt += f"User: {user_in}\nAssistant:"
            tokens = torch.tensor([encode(prompt)], device=device)

            # Generate
            print("\033[92mSARAN:\033[0m ", end="", flush=True)
            response = ""
            for tok_id in model.generate(
                tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty
            ):
                chunk = decode([tok_id])
                response += chunk

                # Check stop sequences
                stopped = False
                for stop in stop_seqs:
                    if stop in response:
                        response = response.split(stop)[0]
                        stopped = True
                        break
                if stopped:
                    break

            response = clean_response(response)
            print(response)
            print()

            history.append({"u": user_in, "a": response})
            # Keep history manageable
            if len(history) > 10:
                history.pop(0)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Error: {e}]\n")

    print("\nGoodbye!")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Device: {device} | Mixed precision: {use_amp}")

    model = SARANMLV(vocab_size, n_embd, block_size, n_layer).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load weights
    if os.path.exists("saran_mlv_ft_best.pt"):
        ckpt = torch.load(
            "saran_mlv_ft_best.pt", map_location=device, weights_only=False
        )
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        val_loss = ckpt.get("best_val_loss", "?")
        print(f"Loaded saran_mlv_ft_best.pt (val_loss={val_loss})")
    else:
        print("ERROR: No model found. Run saran_mlv.py then saran_mlv_ft.py first.")
        sys.exit(1)

    model.eval()
    chat_loop(model)
