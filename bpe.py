"""Byte-level BPE with exact GPT-2 pretokenization and incremental input."""

from __future__ import annotations

import heapq
import json
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKEN = regex.compile(PAT)
_BYTES = tuple(bytes([i]) for i in range(256))
CACHE_CAPACITY = 1_048_576
_STREAM_CACHE_CAPACITY = 2048


class _DescendingPair(tuple):
    """Make heapq prefer the lexicographically largest bytes pair on ties."""

    def __lt__(self, other):
        return tuple.__gt__(self, other)


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab) if vocab is not None else {}
        self.merges = list(merges or [])
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self._rebuild()

    def _rebuild(self):
        if any(not isinstance(s, str) or not s for s in self.special_tokens):
            raise ValueError("Special tokens must be nonempty strings")
        if any(not isinstance(v, bytes) for v in self.vocab.values()):
            raise TypeError("Vocabulary values must be bytes")
        self._ids = {value: key for key, value in self.vocab.items()}
        for token in self.special_tokens:
            raw = token.encode("utf-8")
            if raw not in self._ids:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = raw
                self._ids[raw] = token_id
        self._special_ids = {s: self._ids[s.encode("utf-8")] for s in self.special_tokens}
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("|".join(map(regex.escape, ordered))) if ordered else None
        self._prefixes = {s[:i] for s in ordered for i in range(1, len(s))}
        self._max_special = max(map(len, ordered), default=0)
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache: dict[str, tuple[int, ...]] = {}

    def _pieces(self, text):
        """Yield (special, start, end), isolating specials before regex matching."""
        offset = 0
        if self._special_re:
            for special in self._special_re.finditer(text):
                for match in _PRETOKEN.finditer(text, offset, special.start()):
                    yield False, match.start(), match.end()
                yield True, special.start(), special.end()
                offset = special.end()
        for match in _PRETOKEN.finditer(text, offset):
            yield False, match.start(), match.end()

    def _stream_pieces(self, iterable):
        buffer = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("Input chunks must be strings")
            # Bound temporary storage even if the caller supplies very large chunks.
            for start in range(0, len(chunk), 65536):
                buffer += chunk[start : start + 65536]
                safe_end = len(buffer)
                for size in range(1, min(len(buffer), self._max_special - 1) + 1):
                    if buffer[-size:] in self._prefixes:
                        safe_end = len(buffer) - size
                pending = deque()
                consumed = 0
                # Retain three matches: extending input can change trailing whitespace
                # and an unfinished contraction (apostrophe followed by letters).
                # An incomplete special may also change the preceding whitespace
                # match, so tokenize only the prefix before it.
                for piece in self._pieces(buffer[:safe_end]):
                    pending.append(piece)
                    if len(pending) > 3:
                        special, lo, hi = pending.popleft()
                        if hi > safe_end:
                            break
                        yield special, buffer[lo:hi]
                        consumed = hi
                buffer = buffer[consumed:]
        for special, lo, hi in self._pieces(buffer):
            yield special, buffer[lo:hi]

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:
        specials = list(dict.fromkeys(special_tokens))
        if vocab_size < 256 + len(specials):
            raise ValueError("vocab_size must accommodate 256 bytes and all special tokens")
        self.vocab = dict(enumerate(_BYTES))
        self.merges = []
        self.special_tokens = specials
        self._rebuild()
        frequencies = Counter()
        with open(input_path, encoding="utf-8", newline="") as source:
            chunks = iter(lambda: source.read(1024 * 1024), "")
            for special, piece in self._stream_pieces(chunks):
                if not special:
                    frequencies[piece.encode("utf-8")] += 1
        words = [tuple(_BYTES[b] for b in word) for word in frequencies]
        weights = list(frequencies.values())
        del frequencies
        counts = Counter()
        locations = {}
        for index, word in enumerate(words):
            for pair, count in Counter(zip(word, word[1:])).items():
                counts[pair] += count * weights[index]
                locations.setdefault(pair, set()).add(index)
        heap = [(-count, _DescendingPair(pair)) for pair, count in counts.items()]
        heapq.heapify(heap)
        while len(self.vocab) < vocab_size and heap:
            negative_count, candidate = heapq.heappop(heap)
            pair = tuple(candidate)
            if counts.get(pair, 0) != -negative_count:
                continue
            merged = pair[0] + pair[1]
            touched = set()
            for index in list(locations[pair]):
                word = words[index]
                before = Counter(zip(word, word[1:]))
                result = []
                pos = 0
                while pos < len(word):
                    if pos + 1 < len(word) and (word[pos], word[pos + 1]) == pair:
                        result.append(merged)
                        pos += 2
                    else:
                        result.append(word[pos])
                        pos += 1
                words[index] = tuple(result)
                after = Counter(zip(result, result[1:]))
                for changed in before.keys() | after.keys():
                    delta = after[changed] - before[changed]
                    if delta:
                        counts[changed] += delta * weights[index]
                        touched.add(changed)
                    if changed not in after:
                        locations[changed].discard(index)
                    elif changed not in before:
                        locations.setdefault(changed, set()).add(index)
            self.merges.append(pair)
            if merged not in self._ids:
                token_id = len(self.vocab)
                self.vocab[token_id] = merged
                self._ids[merged] = token_id
            for changed in touched:
                if counts[changed]:
                    heapq.heappush(heap, (-counts[changed], _DescendingPair(changed)))
                else:
                    del counts[changed]
                    locations.pop(changed, None)
            if len(heap) > 4 * len(counts) + 1000:
                heap = [(-count, _DescendingPair(p)) for p, count in counts.items()]
                heapq.heapify(heap)
        self._rebuild()

    def _encode_piece(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        return self._encode_uncached(piece, _STREAM_CACHE_CAPACITY)

    def _encode_uncached(self, piece, cache_capacity=CACHE_CAPACITY):
        """Apply merge ranks to a cache miss, then store eligible results."""
        symbols = [_BYTES[b] for b in piece.encode("utf-8")]
        while len(symbols) > 1:
            best = None
            best_rank = len(self.merges)
            for pair in zip(symbols, symbols[1:]):
                rank = self._ranks.get(pair, best_rank)
                if rank < best_rank:
                    best, best_rank = pair, rank
            if best is None:
                break
            result = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == best:
                    result.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    result.append(symbols[index])
                    index += 1
            symbols = result
        ids = tuple(self._ids[symbol] for symbol in symbols)
        # A small bounded cache also respects the official streaming memory test.
        if len(piece) <= 128:
            if len(self._cache) >= cache_capacity:
                self._cache.clear()
            self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        extend = ids.extend
        encode_piece = self._encode_uncached
        lookup = self._cache.get
        offset = 0
        if self._special_re:
            for special in self._special_re.finditer(text):
                for piece in _PRETOKEN.findall(text, offset, special.start()):
                    cached = lookup(piece)
                    if cached is None:
                        cached = encode_piece(piece)
                    extend(cached)
                ids.append(self._special_ids[special.group()])
                offset = special.end()
        for piece in _PRETOKEN.findall(text, offset):
            cached = lookup(piece)
            if cached is None:
                cached = encode_piece(piece)
            extend(cached)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # The streaming API keeps its working memory small, using the same
        # clear-all cache logic with a private bound rather than a policy switch.
        if len(self._cache) > _STREAM_CACHE_CAPACITY:
            self._cache.clear()
        for special, piece in self._stream_pieces(iterable):
            if special:
                yield self._special_ids[piece]
            else:
                yield from self._encode_piece(piece)

    def decode(self, ids: list[int]) -> str:
        return b"".join(map(self.vocab.__getitem__, ids)).decode("utf-8", errors="replace")

    def save(self, path_prefix: str) -> None:
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        payloads = {
            ".vocab.json": {str(k): v.hex() for k, v in self.vocab.items()},
            ".merges.json": [[a.hex(), b.hex()] for a, b in self.merges],
            ".config.json": {
                "version": 1,
                "encoding": "hex",
                "special_tokens": self.special_tokens,
                "special_token_ids": self._special_ids,
            },
        }
        for suffix, payload in payloads.items():
            Path(str(prefix) + suffix).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path_prefix: str) -> None:
        config = json.loads(Path(str(path_prefix) + ".config.json").read_text(encoding="utf-8"))
        if config.get("version") != 1 or config.get("encoding") != "hex":
            raise ValueError("Unsupported tokenizer serialization format")
        loaded = self.from_files(
            str(path_prefix) + ".vocab.json", str(path_prefix) + ".merges.json", config["special_tokens"]
        )
        if loaded._special_ids != config["special_token_ids"]:
            raise ValueError("Special token IDs do not match the saved configuration")
        self.__dict__.update(loaded.__dict__)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Read this implementation's hex JSON files; config is loaded by load()."""
        vocab = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        return cls(
            {int(k): bytes.fromhex(v) for k, v in vocab.items()},
            [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges],
            special_tokens,
        )
