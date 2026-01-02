import os
import json
import numpy as np
from datasets import load_dataset
import tiktoken

block_size = 256  # minimum document length in tokens
tokens_file = "openwebtext_tokens.jsonl"
offsets_file = "openwebtext_offsets.npy"
_dataset_name = "Skylion007/openwebtext"

enc = tiktoken.get_encoding("gpt2")

if not (os.path.exists(tokens_file) and os.path.exists(offsets_file)):
    print(
        "Tokenizing (streaming) and writing tokenized examples to disk...", flush=True
    )
    offsets = []
    buf = []
    buf_bytes = 0
    pos = 0
    buf_max = 1000
    with open(tokens_file, "wb") as fw:
        for i, ex in enumerate(
            load_dataset(_dataset_name, split="train", streaming=True), start=1
        ):
            txt = ex.get("text")
            if not txt:
                continue
            toks = enc.encode(txt)
            if len(toks) < block_size + 1:
                continue
            line = json.dumps(toks).encode("utf-8") + b"\n"
            offsets.append(pos + buf_bytes)
            buf.append(line)
            buf_bytes += len(line)
            if len(buf) >= buf_max:
                fw.write(b"".join(buf))
                pos += buf_bytes
                buf = []
                buf_bytes = 0
            if i % 1000 == 0:
                fw.flush()
                print(
                    f"tokenized docs: {i:,}, written bytes ~{pos + buf_bytes:,}",
                    flush=True,
                )
        if buf:
            fw.write(b"".join(buf))
            pos += buf_bytes
    np.save(offsets_file, np.array(offsets, dtype=np.int64))
    print(f"Done tokenizing. Wrote {len(offsets)} examples, bytes ~{pos}", flush=True)
else:
    print(f"Dataset files already exist:")
    print(f"  - {tokens_file}")
    print(f"  - {offsets_file}")
