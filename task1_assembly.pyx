# cython: language_level=3, boundscheck=False, wraparound=False
"""Checked uint32 boxing and diagnostic-only decode; no tokenizer algorithms."""
from cpython.object cimport PyObject
from cpython.list cimport PyList_New
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.long cimport PyLong_AsSsize_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport uint32_t, uintptr_t
from libc.string cimport memcpy
import time

cdef extern from "Python.h":
    PyObject *new_integer "PyLong_FromUnsignedLong"(unsigned long value) except NULL
    PyObject *tuple_item "PyTuple_GET_ITEM"(PyObject *value, Py_ssize_t index)
    void incref "Py_INCREF"(PyObject *value)
    void set_list_item "PyList_SET_ITEM"(PyObject *value, Py_ssize_t index, PyObject *item)


def _checked_list_size(Py_ssize_t count):
    if count < 0 or count > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(PyObject *):
        raise OverflowError('List pointer buffer size overflow')
    return count


cdef class Boxer:
    cdef Py_ssize_t vocabulary_size
    cdef bint reuse
    cdef tuple pool

    def __init__(self, Py_ssize_t vocabulary_size, bint reuse=False):
        _checked_list_size(vocabulary_size)
        if vocabulary_size > 0x100000000:
            raise OverflowError('Vocabulary is outside uint32 range')
        self.vocabulary_size = vocabulary_size
        self.reuse = reuse
        self.pool = tuple(range(vocabulary_size)) if reuse else ()

    def convert(self, buffer):
        if buffer is None:
            raise TypeError('Expected a uint32 buffer')
        cdef const uint32_t[::1] values = buffer
        cdef Py_ssize_t n = values.shape[0], i
        cdef uint32_t index
        cdef PyObject *item
        cdef list result
        _checked_list_size(n)
        if n and <uintptr_t>&values[0] % sizeof(uint32_t):
            raise BufferError('Unaligned uint32 buffer')
        result = PyList_New(n)
        for i in range(n):
            index = values[i]
            if index >= self.vocabulary_size:
                raise ValueError('Worker ID is outside the vocabulary')
            if self.reuse:
                item = tuple_item(<PyObject *>self.pool, index)
                incref(item)  # New list steals exactly this new reference.
            else:
                item = new_integer(index)  # New reference, normal CPython int policy.
            set_list_item(<PyObject *>result, i, item)
        return result


def profile_decode(vocab, ids):
    """Diagnostic replica of the existing decoder, never used for ranking."""
    cdef Py_ssize_t n, count, i, index, total = 0, position = 0
    cdef const char **pointers = NULL
    cdef Py_ssize_t *lengths = NULL
    cdef list refs
    cdef object token_id, value, result
    cdef bytes raw
    cdef char *destination
    cdef double began, started
    began = time.perf_counter()
    metrics = {'backend': 'native', 'diagnostic_implementation': 'assembly-profile',
               'table_seconds': 0., 'validate_length_seconds': 0., 'allocation_seconds': 0.,
               'copy_first_touch_seconds': 0., 'utf8_seconds': 0., 'cleanup_seconds': 0.}
    if type(vocab) is not dict or type(ids) is not list:
        raise TypeError('Profile requires ordinary dict and list')
    if not ids:
        metrics['total_seconds'] = time.perf_counter() - began
        return '', metrics
    n, count = len(vocab), len(ids)
    if not n or n > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(char *) or n > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(Py_ssize_t):
        raise OverflowError('Profile vocabulary pointer size overflow')
    started = time.perf_counter()
    refs = [vocab[i] for i in range(n)]
    if any(type(value) is not bytes for value in refs):
        raise TypeError('Profile requires bytes vocabulary values')
    try:
        pointers = <const char **>PyMem_Malloc(n * sizeof(char *))
        lengths = <Py_ssize_t *>PyMem_Malloc(n * sizeof(Py_ssize_t))
        if pointers == NULL or lengths == NULL:
            raise MemoryError('Cannot allocate vocabulary pointer tables')
        for i in range(n):
            value = refs[i]
            pointers[i] = PyBytes_AS_STRING(value)
            lengths[i] = PyBytes_GET_SIZE(value)
        metrics['table_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        for i in range(count):
            token_id = ids[i]
            if type(token_id) is not int:
                raise TypeError('Profile requires exact integer IDs')
            try:
                index = PyLong_AsSsize_t(token_id)
            except OverflowError:
                raise KeyError(token_id)
            if index < 0 or index >= n:
                raise KeyError(token_id)
            if lengths[index] > PY_SSIZE_T_MAX - total:
                raise OverflowError('Decoded byte length exceeds Py_ssize_t')
            total += lengths[index]
        metrics['validate_length_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        raw = PyBytes_FromStringAndSize(NULL, total)
        destination = PyBytes_AS_STRING(raw)
        metrics['allocation_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        for i in range(count):
            index = PyLong_AsSsize_t(ids[i])
            if lengths[index]:
                memcpy(destination + position, pointers[index], lengths[index])
            position += lengths[index]
        metrics['copy_first_touch_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        result = raw.decode('utf-8', errors='replace')
        metrics['utf8_seconds'] = time.perf_counter() - started
        metrics['output_bytes'] = total
    finally:
        started = time.perf_counter()
        PyMem_Free(pointers)
        PyMem_Free(lengths)
        raw = b''
        refs.clear()
        metrics['cleanup_seconds'] = time.perf_counter() - started
    metrics['total_seconds'] = time.perf_counter() - began
    return result, metrics
