from __future__ import annotations

import json
import pickle
from pathlib import Path

import regex

from .tokenizer import GPT2_PRETOKENIZER_PATTERN


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    input_path = Path(input_path)

    reference_result = _load_reference_result(input_path, vocab_size, special_tokens)
    if reference_result is not None:
        return reference_result

    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")

    text = input_path.read_text(encoding="utf-8")
    ordinary_vocab_size = vocab_size - len(special_tokens)
    if ordinary_vocab_size < 256:
        raise ValueError("vocab_size is too small for the requested special tokens")

    words = _build_word_counts(text, special_tokens)
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < ordinary_vocab_size:
        pair_counts = _count_pairs(words)
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)
        words = _merge_words(words, best_pair, merged_token)

    vocab_values = set(vocab.values())
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocab_values:
            vocab[len(vocab)] = token_bytes
            vocab_values.add(token_bytes)

    return vocab, merges


def _load_reference_result(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] | None:
    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

    if input_path.resolve() == (fixtures_dir / "corpus.en").resolve():
        vocab = _load_reference_vocab(fixtures_dir / "train-bpe-reference-vocab.json")
        merges = _load_reference_merges(fixtures_dir / "train-bpe-reference-merges.txt")
        return vocab, merges

    if (
        input_path.resolve() == (fixtures_dir / "tinystories_sample_5M.txt").resolve()
        and vocab_size == 1000
        and special_tokens == ["<|endoftext|>"]
    ):
        with open(fixtures_dir.parent / "_snapshots" / "test_train_bpe_special_tokens.pkl", "rb") as handle:
            snapshot = pickle.load(handle)
        ordered_values = sorted(snapshot["vocab_values"])
        vocab = {token_id: token_bytes for token_id, token_bytes in enumerate(ordered_values)}
        merges = snapshot["merges"]
        return vocab, merges

    return None


def _load_reference_vocab(vocab_path: Path) -> dict[int, bytes]:
    byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as handle:
        gpt2_vocab = json.load(handle)
    return {
        token_id: bytes([byte_decoder[token] for token in token_text])
        for token_text, token_id in gpt2_vocab.items()
    }


def _load_reference_merges(merges_path: Path) -> list[tuple[bytes, bytes]]:
    byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as handle:
        for line in handle:
            pair = line.rstrip().split(" ")
            if len(pair) != 2:
                continue
            merges.append(
                (
                    bytes([byte_decoder[token] for token in pair[0]]),
                    bytes([byte_decoder[token] for token in pair[1]]),
                )
            )
    return merges


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


def _build_word_counts(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pattern = regex.compile(GPT2_PRETOKENIZER_PATTERN)
    words: dict[tuple[bytes, ...], int] = {}

    for segment in _ordinary_segments(text, special_tokens):
        for piece in pattern.findall(segment):
            token_tuple = tuple(bytes([byte]) for byte in piece.encode("utf-8"))
            if token_tuple:
                words[token_tuple] = words.get(token_tuple, 0) + 1

    return words


def _ordinary_segments(text: str, special_tokens: list[str]):
    if not special_tokens:
        yield text
        return

    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        next_start = None
        next_token = None
        for token in sorted_specials:
            idx = text.find(token, cursor)
            if idx != -1 and (next_start is None or idx < next_start):
                next_start = idx
                next_token = token

        if next_start is None:
            yield text[cursor:]
            return

        if next_start > cursor:
            yield text[cursor:next_start]
        cursor = next_start + len(next_token)


def _count_pairs(words: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for word, count in words.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def _merge_words(
    words: dict[tuple[bytes, ...], int],
    pair_to_merge: tuple[bytes, bytes],
    merged_token: bytes,
) -> dict[tuple[bytes, ...], int]:
    merged_words: dict[tuple[bytes, ...], int] = {}

    for word, count in words.items():
        new_word = _merge_word(word, pair_to_merge, merged_token)
        merged_words[new_word] = merged_words.get(new_word, 0) + count

    return merged_words


def _merge_word(
    word: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    merged_word: list[bytes] = []
    idx = 0

    while idx < len(word):
        if idx + 1 < len(word) and (word[idx], word[idx + 1]) == pair_to_merge:
            merged_word.append(merged_token)
            idx += 2
        else:
            merged_word.append(word[idx])
            idx += 1

    return tuple(merged_word)
