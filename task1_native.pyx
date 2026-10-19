# cython: language_level=3, boundscheck=False, wraparound=False
"""Own BPE helpers: checked, GIL-held loops; no tokenizer dependency."""
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.long cimport PyLong_AsSsize_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport uint64_t, uint32_t
from libc.string cimport memcpy, memset
from libc.stddef cimport size_t


def _checked_total(Py_ssize_t total, Py_ssize_t length):
    """Same checked addition used by decode; exposed for overflow regression."""
    if total < 0 or length < 0 or length > PY_SSIZE_T_MAX - total:
        raise OverflowError('Decoded byte length exceeds Py_ssize_t')
    return total + length


cdef object fallback(object vocab, object ids, object stats):
    from bpe_decode_runtime import serial
    if stats is not None:
        stats['backend'] = 'python-fallback'
    return serial(vocab, ids)


def decode(vocab, ids, stats=None):
    """Two checked passes over list[int], one bytes allocation, one UTF-8 decode."""
    cdef Py_ssize_t n, count, i, index, total = 0, position = 0
    cdef const char **pointers = NULL
    cdef Py_ssize_t *lengths = NULL
    cdef list refs
    cdef object token_id, value
    cdef bytes raw
    cdef char *destination
    if type(vocab) is not dict or type(ids) is not list:
        return fallback(vocab, ids, stats)
    if not ids:
        if stats is not None:
            stats['backend'] = 'native'
        return ''
    n, count = len(vocab), len(ids)
    if not n or n > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(Py_ssize_t) or n > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(char *):
        return fallback(vocab, ids, stats)
    # Building fresh tables makes mutations between calls visible, without a
    # vocabulary scan or table construction before the timed decode call.
    try:
        refs = [vocab[i] for i in range(n)]
    except KeyError:
        return fallback(vocab, ids, stats)
    if any(type(value) is not bytes for value in refs):
        return fallback(vocab, ids, stats)
    try:
        pointers = <const char **>PyMem_Malloc(n * sizeof(char *))
        lengths = <Py_ssize_t *>PyMem_Malloc(n * sizeof(Py_ssize_t))
        if pointers == NULL or lengths == NULL:
            raise MemoryError('Cannot allocate vocabulary pointer tables')
        for i in range(n):
            value = refs[i]
            pointers[i] = PyBytes_AS_STRING(value)
            lengths[i] = PyBytes_GET_SIZE(value)
        for i in range(count):
            token_id = ids[i]
            if type(token_id) is not int:
                return fallback(vocab, ids, stats)
            try:
                index = PyLong_AsSsize_t(token_id)
            except OverflowError:
                # Preserve Python dictionary KeyError rather than conversion error.
                return fallback(vocab, ids, stats)
            if index < 0 or index >= n:
                raise KeyError(token_id)
            if lengths[index] > PY_SSIZE_T_MAX - total:
                raise OverflowError('Decoded byte length exceeds Py_ssize_t')
            total += lengths[index]
        raw = PyBytes_FromStringAndSize(NULL, total)
        destination = PyBytes_AS_STRING(raw)
        for i in range(count):
            index = PyLong_AsSsize_t(ids[i])
            if lengths[index]:
                memcpy(destination + position, pointers[index], lengths[index])
            position += lengths[index]
        if stats is not None:
            stats.update({'backend': 'native', 'output_bytes': total, 'vocabulary_entries': n})
        return raw.decode('utf-8', errors='replace')
    finally:
        PyMem_Free(pointers)
        PyMem_Free(lengths)


class UnsupportedEncoding(ValueError):
    pass


cdef struct PairEntry:
    uint64_t key
    Py_ssize_t rank
    uint32_t merged
    unsigned char used


cdef uint64_t pair_key(uint32_t left, uint32_t right) noexcept:
    return (<uint64_t>left << 32) | right


cdef uint64_t mix(uint64_t value) noexcept:
    value ^= value >> 30
    value *= <uint64_t>0xbf58476d1ce4e5b9
    value ^= value >> 27
    value *= <uint64_t>0x94d049bb133111eb
    return value ^ (value >> 31)


cdef class Encoder:
    cdef PairEntry *table
    cdef Py_ssize_t capacity, rank_limit
    cdef list symbols, final_ids
    cdef object missing

    def __cinit__(self):
        self.table = NULL
        self.capacity = 0

    def __init__(self, ranks, ids, rank_limit):
        cdef dict lookup
        cdef object pair, rank, left, right, merged, symbol
        cdef uint32_t a, b, result
        cdef Py_ssize_t slot, capacity = 2
        cdef uint64_t key
        if type(ranks) is not dict or type(ids) is not dict or type(rank_limit) is not int or rank_limit < 0:
            raise UnsupportedEncoding('Requires ordinary BPE rank/ID dictionaries')
        if len(ranks) > PY_SSIZE_T_MAX // <Py_ssize_t>(2 * sizeof(PairEntry)):
            raise OverflowError('Merge table too large')
        while capacity < 2 * len(ranks):
            capacity *= 2
        if capacity > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(PairEntry):
            raise OverflowError('Merge table allocation overflow')
        self.rank_limit = rank_limit
        self.capacity = capacity
        self.table = <PairEntry *>PyMem_Malloc(capacity * sizeof(PairEntry))
        if self.table == NULL:
            raise MemoryError('Cannot allocate merge rank table')
        memset(self.table, 0, capacity * sizeof(PairEntry))
        self.symbols = [bytes([i]) for i in range(256)]
        lookup = {raw: i for i, raw in enumerate(self.symbols)}
        for pair, rank in ranks.items():
            if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not bytes
                    or type(pair[1]) is not bytes or type(rank) is not int or not 0 <= rank < rank_limit):
                raise UnsupportedEncoding('Nonstandard merge map')
            left, right = pair
            merged = left + right
            for symbol in (left, right, merged):
                if symbol not in lookup:
                    if len(self.symbols) >= 0xffffffff:
                        raise OverflowError('Too many internal symbols')
                    lookup[symbol] = len(self.symbols)
                    self.symbols.append(symbol)
            a, b, result = lookup[left], lookup[right], lookup[merged]
            key = pair_key(a, b)
            slot = mix(key) & (capacity - 1)
            while self.table[slot].used and self.table[slot].key != key:
                slot = (slot + 1) & (capacity - 1)
            self.table[slot].used = 1
            self.table[slot].key = key
            self.table[slot].rank = rank
            self.table[slot].merged = result
        self.missing = object()
        self.final_ids = [ids.get(raw, self.missing) for raw in self.symbols]

    def __dealloc__(self):
        PyMem_Free(self.table)

    cdef Py_ssize_t find(self, uint64_t key) noexcept:
        cdef Py_ssize_t slot = mix(key) & (self.capacity - 1)
        while self.table[slot].used:
            if self.table[slot].key == key:
                return slot
            slot = (slot + 1) & (self.capacity - 1)
        return -1

    def encode(self, bytes raw):
        cdef Py_ssize_t n = len(raw), i, out, slot, best_slot, best_rank
        cdef uint64_t best_key
        cdef uint32_t *values = NULL
        cdef const unsigned char *data = <const unsigned char *>PyBytes_AS_STRING(raw)
        cdef list result
        cdef object token_id
        if n > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(uint32_t):
            raise OverflowError('Piece buffer size overflow')
        if not n:
            return ()
        try:
            values = <uint32_t *>PyMem_Malloc(n * sizeof(uint32_t))
            if values == NULL:
                raise MemoryError('Cannot allocate piece buffer')
            for i in range(n):
                values[i] = data[i]
            while n > 1:
                best_slot, best_rank = -1, self.rank_limit
                for i in range(n - 1):
                    slot = self.find(pair_key(values[i], values[i + 1]))
                    if slot >= 0 and self.table[slot].rank < best_rank:
                        best_slot, best_rank = slot, self.table[slot].rank
                if best_slot < 0:
                    break
                best_key = self.table[best_slot].key
                i, out = 0, 0
                while i < n:
                    if i + 1 < n and pair_key(values[i], values[i + 1]) == best_key:
                        values[out] = self.table[best_slot].merged
                        i += 2
                    else:
                        values[out] = values[i]
                        i += 1
                    out += 1
                n = out
            result = []
            for i in range(n):
                token_id = self.final_ids[values[i]]
                if token_id is self.missing:
                    raise KeyError(self.symbols[values[i]])
                result.append(token_id)
            return tuple(result)
        finally:
            PyMem_Free(values)
