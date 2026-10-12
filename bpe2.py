from __future__ import annotations

import heapq
import json
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKEN=regex.compile(PAT, regex.V1)
_BYTE=tuple(bytes([i]) for i in range(256))
CACHE_CAPACITY = 2**20  # 1M
_STREAM_CACHE_CAPACITY = 2048  # 2K



class _DescendingPair(tuple):

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
        if any(not isinstance(s, str) for s in self.special_tokens):
            raise TypeError("special_tokens 的值必须是字符串")
        if any(not isinstance(v, bytes) for v in self.vocab.values()):
            raise TypeError("vocab 的值必须是 bytes")
        self._ids = {v: k for k, v in self.vocab.items()}
        for token in self.special_tokens:
            raw = token.encode("utf-8")
            if raw not in self._ids:
                token_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[token_id] = raw
                self._ids[raw] = token_id 
        self._special_ids = {s: self._ids[s.encode("utf-8")] for s in self.special_tokens}
        ordered = sorted(self.special_tokens,key=len,reverse=True)
        self._special_re = regex.compile("|".join(map(regex.escape, ordered))) if ordered else None
        self.prefixes= {v[:i] for v in self._ids for i in range(1, len(v))}
        self._max_special =max(map(len,ordered),default=0)
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: dict[str, tuple[int, ...]] = {}
        
    def _pieces(self, text: str) -> Iterator[bytes]:
        """先隔离特殊 token，再匹配普通文本；返回特殊标志及起止位置。"""
        offset = 0
        if self._special_re:
            for special in self._special_re.finditer(text):
                for match in _PRETOKEN.finditer(text,offset:special.start()):
                    yield False, match.start(), match.end()
                yield True, special.start(), special.end()
                offset = special.end()
            for match in _PRETOKEN.finditer(text,offset):
                yield False, match.start(), match.end()
        
    def _stream_pieces(self,iterables):
        buffer = ""
        for chunk in iterables:
            if not isinstance(chunk, str):
                raise TypeError("iterables 的值必须是字符串")
            for start in range(0, len(chunk), _STREAM_CACHE_CAPACITY):
                buffer += chunk[start:start + _STREAM_CACHE_CAPACITY]
                safe_end =len(buffer)
                for size in range(1, min(len(buffer), self._max_special) + 1):
                    if buffer[-size:] in self._prefixes:
                        safe_end = len(buffer) - size
                    pending = deque()
                    consumed = 0
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
            
    def train(self, input_path: str, vocab_size: int, special_tokens: list[str] ) -> None:
        """训练 BPE 模型。"""
        if not isinstance(input_path, str):
            raise TypeError("input_path 的值必须是字符串")
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size 的值必须是整数")
        if not isinstance(special_tokens, list) or any(not isinstance(s, str) for s in special_tokens):
            raise TypeError("special_tokens 的值必须是字符串列表")
        self.special_tokens = list(dict.fromkeys(special_tokens))
        self._rebuild()
        counter = Counter()
        with open(input_path, encoding="utf-8", newline="") as f:
            for special, piece in self._stream_pieces(f):
                if special:
                    continue
                counter[piece.encode("utf-8")] += 1
        vocab = {i: b for i, (b, _) in enumerate(counter.most_common(vocab_size))}
        self.vocab = vocab
        self.merges = []
        self._rebuild()
    
                                  