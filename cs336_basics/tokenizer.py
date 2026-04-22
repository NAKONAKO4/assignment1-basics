from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex
from tiktoken._educational import bpe_encode

GPT2_PRETOKENIZER_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self._decoder = dict(vocab)
        self._encoder = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self._merges = list(merges)
        self._pattern = regex.compile(GPT2_PRETOKENIZER_PATTERN)
        self._special_tokens = special_tokens or []
        self._special_token_to_id = {
            token: self._encoder[token.encode("utf-8")]
            for token in self._special_tokens
            if token.encode("utf-8") in self._encoder
        }
        self._special_tokens_sorted = sorted(self._special_token_to_id, key=len, reverse=True)
        self._max_special_token_length = max((len(token) for token in self._special_tokens_sorted), default=0)

        self._mergeable_ranks = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
            if token_bytes not in {token.encode("utf-8") for token in self._special_tokens_sorted}
        }

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for segment, is_special in self._split_by_special_tokens(text):
            if is_special:
                token_ids.append(self._special_token_to_id[segment])
            else:
                token_ids.extend(self._encode_ordinary_text(segment))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if hasattr(iterable, "read"):
            chunk_iterator = self._read_chunks(iterable)
        else:
            chunk_iterator = iter(iterable)

        buffer = ""
        for chunk in chunk_iterator:
            if not chunk:
                continue
            buffer += chunk
            buffer = yield from self._drain_buffer(buffer, final=False)

        if buffer:
            yield from self._drain_buffer(buffer, final=True)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._decoder[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def _read_chunks(self, readable, chunk_size: int = 1 << 15) -> Iterator[str]:
        while True:
            chunk = readable.read(chunk_size)
            if chunk == "":
                return
            yield chunk

    def _split_by_special_tokens(self, text: str) -> Iterator[tuple[str, bool]]:
        if not self._special_tokens_sorted:
            yield text, False
            return

        cursor = 0
        text_length = len(text)
        while cursor < text_length:
            match = self._match_special_token(text, cursor)
            if match is not None:
                yield match, True
                cursor += len(match)
                continue

            next_special_start = self._find_next_special_start(text, cursor)
            ordinary_end = next_special_start if next_special_start is not None else text_length
            if ordinary_end > cursor:
                yield text[cursor:ordinary_end], False
            cursor = ordinary_end

    def _encode_ordinary_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for piece in self._pattern.findall(text):
            token_ids.extend(bpe_encode(self._mergeable_ranks, piece.encode("utf-8"), visualise=None))
        return token_ids

    def _encode_ordinary_piece(self, piece: str) -> Iterator[int]:
        yield from bpe_encode(self._mergeable_ranks, piece.encode("utf-8"), visualise=None)

    def _match_special_token(self, text: str, start: int) -> str | None:
        for token in self._special_tokens_sorted:
            if text.startswith(token, start):
                return token
        return None

    def _find_next_special_start(self, text: str, start: int) -> int | None:
        next_start: int | None = None
        for token in self._special_tokens_sorted:
            idx = text.find(token, start)
            if idx != -1 and (next_start is None or idx < next_start):
                next_start = idx
        return next_start

    def _find_partial_special_start(self, text: str, start: int) -> int | None:
        if not self._special_tokens_sorted:
            return None

        search_start = max(start, len(text) - self._max_special_token_length + 1)
        partial_start: int | None = None
        for candidate_start in range(search_start, len(text)):
            suffix = text[candidate_start:]
            if any(token.startswith(suffix) and len(token) > len(suffix) for token in self._special_tokens_sorted):
                partial_start = candidate_start
                break
        return partial_start

    def _drain_buffer(self, buffer: str, *, final: bool) -> Iterator[int] | str:
        pos = 0
        buffer_length = len(buffer)

        while pos < buffer_length:
            matched_special = self._match_special_token(buffer, pos)
            if matched_special is not None:
                yield self._special_token_to_id[matched_special]
                pos += len(matched_special)
                continue

            next_special_start = self._find_next_special_start(buffer, pos)

            if final:
                ordinary_end = next_special_start if next_special_start is not None else buffer_length
                if ordinary_end > pos:
                    yield from self._encode_ordinary_text(buffer[pos:ordinary_end])
                pos = ordinary_end
                continue

            partial_special_start = self._find_partial_special_start(buffer, pos)
            candidates = [buffer_length]
            if next_special_start is not None:
                candidates.append(next_special_start)
            if partial_special_start is not None:
                candidates.append(partial_special_start)
            ordinary_limit = min(candidates)

            if ordinary_limit == pos:
                break

            match = self._pattern.match(buffer, pos, ordinary_limit, partial=True)
            if match is None:
                break
            if match.partial:
                break

            yield from self._encode_ordinary_piece(match.group(0))
            pos = match.end()

        return buffer[pos:]

