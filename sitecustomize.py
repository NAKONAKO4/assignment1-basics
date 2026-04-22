from __future__ import annotations

import json
from pathlib import Path

import tiktoken.registry as registry
import tiktoken_ext.openai_public as openai_public


def _local_gpt2():
    project_root = Path(__file__).resolve().parent
    fixtures_dir = project_root / "tests" / "fixtures"
    with open(fixtures_dir / "gpt2_vocab.json", encoding="utf-8") as handle:
        raw_vocab = json.load(handle)
    byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
    mergeable_ranks = {
        bytes([byte_decoder[ch] for ch in token_text]): token_id
        for token_text, token_id in raw_vocab.items()
        if token_id != 50256
    }
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        "pat_str": openai_public.r50k_pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {openai_public.ENDOFTEXT: 50256},
    }


openai_public.gpt2 = _local_gpt2
openai_public.ENCODING_CONSTRUCTORS["gpt2"] = _local_gpt2
registry.ENCODINGS.pop("gpt2", None)
if registry.ENCODING_CONSTRUCTORS is not None:
    registry.ENCODING_CONSTRUCTORS["gpt2"] = _local_gpt2


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(code_point) for code_point in cs)))
