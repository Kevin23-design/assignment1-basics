# cython: language_level=3, boundscheck=False, wraparound=False
"""GIL-held, borrowed-list reads; no private CPython integer representation."""
from cpython.object cimport PyObject
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport uint32_t
from libc.string cimport memcpy
import time

cdef extern from "Python.h":
    PyObject *borrowed_item "PyList_GET_ITEM"(PyObject *values, Py_ssize_t index)
    int exact_integer "PyLong_CheckExact"(PyObject *value)
    Py_ssize_t as_index "PyLong_AsSsize_t"(PyObject *value) noexcept
    PyObject *current_error "PyErr_Occurred"() noexcept
    void clear_error "PyErr_Clear"() noexcept


def _checked_size(Py_ssize_t count, Py_ssize_t width):
    if count < 0 or width <= 0 or count > PY_SSIZE_T_MAX // width:
        raise OverflowError('Native buffer allocation size overflow')
    return count * width


def _checked_total(Py_ssize_t total, Py_ssize_t length):
    if total < 0 or length < 0 or length > PY_SSIZE_T_MAX - total:
        raise OverflowError('Decoded byte length exceeds Py_ssize_t')
    return total + length


cdef void allocation_check(Py_ssize_t size, Py_ssize_t limit) except *:
    if limit >= 0 and size > limit:
        raise MemoryError('Injected allocation limit')


cdef object fallback(object vocab, object ids, object stats):
    from bpe_decode_runtime import serial
    if stats is not None:
        stats['backend'] = 'python-fallback'
    return serial(vocab, ids)


def decode(vocab, ids, bint indexed=False, stats=None):
    return _run(vocab, ids, indexed, stats, False)


def profile_decode(vocab, ids, bint indexed=False, stats=None):
    return _run(vocab, ids, indexed, stats, True)


def _run(vocab, ids, bint indexed=False, stats=None, bint diagnostic=False,
         Py_ssize_t allocation_limit=-1):
    # Exact list + strong argument reference + uninterrupted GIL-held native
    # loops keep borrowed elements alive. No callbacks or private digit access.
    cdef Py_ssize_t n, count, i, index, size, total = 0, position = 0
    cdef const char **pointers = NULL
    cdef Py_ssize_t *lengths = NULL
    cdef uint32_t *indices = NULL
    cdef PyObject *item
    cdef list refs, values
    cdef object value, result
    cdef bytes raw
    cdef char *destination
    cdef double began = 0., tick = 0.
    cdef dict phases = {}
    if diagnostic:
        began = tick = time.perf_counter()
    if type(vocab) is not dict or type(ids) is not list:
        return fallback(vocab, ids, stats)
    values = ids  # One owning reference for the whole call, never per element.
    if not ids:
        if stats is not None:
            stats.update(backend='native', decoder_case='N2' if indexed else 'N1', output_bytes=0, index_buffer_bytes=0)
        return ''
    n, count = len(vocab), len(ids)
    if not n or (indexed and n > 0x100000000):
        return fallback(vocab, ids, stats)
    if any(type(key) is not int for key in vocab):
        return fallback(vocab, ids, stats)
    try:
        refs = [vocab[i] for i in range(n)]
    except KeyError:
        return fallback(vocab, ids, stats)
    if any(type(value) is not bytes for value in refs):
        return fallback(vocab, ids, stats)
    try:
        size = _checked_size(n, sizeof(char *))
        allocation_check(size, allocation_limit)
        pointers = <const char **>PyMem_Malloc(size)
        size = _checked_size(n, sizeof(Py_ssize_t))
        allocation_check(size, allocation_limit)
        lengths = <Py_ssize_t *>PyMem_Malloc(size)
        if pointers == NULL or lengths == NULL:
            raise MemoryError('Cannot allocate vocabulary pointer tables')
        for i in range(n):
            value = refs[i]
            pointers[i] = PyBytes_AS_STRING(value)
            lengths[i] = PyBytes_GET_SIZE(value)
        if diagnostic:
            phases['table_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
        if indexed:
            size = _checked_size(count, sizeof(uint32_t))
            allocation_check(size, allocation_limit)
            indices = <uint32_t *>PyMem_Malloc(size)
            if indices == NULL:
                raise MemoryError('Cannot allocate decode index buffer')
        if diagnostic:
            phases['index_allocation_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
        for i in range(count):
            item = borrowed_item(<PyObject *>values, i)
            if not exact_integer(item):
                return fallback(vocab, ids, stats)
            index = as_index(item)
            # Exact ints can only fail this conversion by overflow. Handle the
            # C error indicator on that path, without a per-item Python try.
            if index == -1 and current_error() != NULL:
                clear_error()
                return fallback(vocab, ids, stats)
            if index < 0 or index >= n:
                raise KeyError(<object>item)
            if lengths[index] > PY_SSIZE_T_MAX - total:
                raise OverflowError('Decoded byte length exceeds Py_ssize_t')
            total += lengths[index]
            if indexed:
                indices[i] = <uint32_t>index
        if diagnostic:
            phases['validate_length_index_seconds'] = time.perf_counter() - tick
            phases['index_write_included'] = bool(indexed)
            tick = time.perf_counter()
        allocation_check(total, allocation_limit)
        raw = PyBytes_FromStringAndSize(NULL, total)
        destination = PyBytes_AS_STRING(raw)
        if diagnostic:
            phases['allocation_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
        # Separate loops remove the method branch from the second hot pass.
        if indexed:
            for i in range(count):
                index = indices[i]
                if lengths[index]:
                    memcpy(destination + position, pointers[index], lengths[index])
                position += lengths[index]
        else:
            for i in range(count):
                index = as_index(borrowed_item(<PyObject *>values, i))
                if lengths[index]:
                    memcpy(destination + position, pointers[index], lengths[index])
                position += lengths[index]
        if diagnostic:
            phases['copy_first_touch_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
        result = raw.decode('utf-8', errors='replace')
        if diagnostic:
            phases['utf8_seconds'] = time.perf_counter() - tick
        if stats is not None:
            stats.update(backend='native', decoder_case='N2' if indexed else 'N1', output_bytes=total,
                         vocabulary_entries=n, index_buffer_bytes=count * sizeof(uint32_t) if indexed else 0)
    finally:
        if diagnostic:
            tick = time.perf_counter()
        PyMem_Free(indices)
        PyMem_Free(pointers)
        PyMem_Free(lengths)
        if diagnostic:
            raw = b''
            refs.clear()
            phases['cleanup_seconds'] = time.perf_counter() - tick
    if diagnostic and stats is not None:
        stats.update(phases, total_seconds=time.perf_counter() - began, diagnostic_implementation='decode-access-profile')
    return result
