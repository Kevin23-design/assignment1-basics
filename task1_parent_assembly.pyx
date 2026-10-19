# cython: language_level=3, boundscheck=False, wraparound=False
"""Pooled direct append / observed-count assembly; no BPE or tokenizer algorithms."""
from array import array
from cpython.object cimport PyObject
from cpython.list cimport PyList_New
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport uint32_t, uintptr_t

cdef extern from "Python.h":
    PyObject *pool_item "PyTuple_GET_ITEM"(PyObject *pool, Py_ssize_t index) noexcept
    void incref "Py_INCREF"(PyObject *value) noexcept
    void set_item "PyList_SET_ITEM"(PyObject *values, Py_ssize_t index, PyObject *value) noexcept
    int append_item "PyList_Append"(PyObject *values, PyObject *value) except -1


def _checked_count(Py_ssize_t current, Py_ssize_t additional=0):
    if current < 0 or additional < 0 or additional > PY_SSIZE_T_MAX - current:
        raise OverflowError('Assembly token count overflow')
    total = current + additional
    if total > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(PyObject *):
        raise OverflowError('Assembly list allocation overflow')
    return total


cdef void check_buffer(object buffer) except *:
    if type(buffer) is not array or buffer.typecode != 'I' or buffer.itemsize != 4:
        raise TypeError('Expected an owned native uint32 array')


cdef class Pool:
    cdef tuple pool
    cdef Py_ssize_t size

    def __init__(self, Py_ssize_t size):
        _checked_count(size)
        if size > 0x100000000:
            raise OverflowError('Vocabulary outside uint32 range')
        self.size = size
        self.pool = tuple(range(size))

    def append(self, target, buffer):
        cdef const uint32_t[::1] values
        cdef Py_ssize_t i, count
        cdef uint32_t index
        if type(target) is not list:
            raise TypeError('Target must be an ordinary list')
        check_buffer(buffer)
        values = buffer
        count = values.shape[0]
        _checked_count(len(target), count)
        if count and <uintptr_t>&values[0] % sizeof(uint32_t):
            raise BufferError('Unaligned uint32 array')
        for i in range(count):
            index = values[i]
            if index >= self.size:
                raise ValueError('Worker ID outside vocabulary')
            # PyList_Append owns the increment; the borrowed pool reference
            # must not be incremented again by this adapter.
            append_item(<PyObject *>target, pool_item(<PyObject *>self.pool, index))
        return count

    def assemble_owned(self, chunks, Py_ssize_t allocation_limit=-1):
        """Consume owned arrays, allocate using observed lengths, never a reference count."""
        cdef const uint32_t[::1] values
        cdef Py_ssize_t part, i, count, total = 0, position = 0
        cdef uint32_t index
        cdef PyObject *item
        cdef list result
        cdef object chunk
        if type(chunks) is not list:
            raise TypeError('Chunks must be an owned ordinary list')
        # Validate all formats and sum actual buffer lengths before allocating.
        for chunk in chunks:
            check_buffer(chunk)
            values = chunk
            count = values.shape[0]
            if count and <uintptr_t>&values[0] % sizeof(uint32_t):
                raise BufferError('Unaligned uint32 array')
            total = _checked_count(total, count)
        values = None
        chunk = None
        if allocation_limit >= 0 and total > allocation_limit:
            raise MemoryError('Injected final-list allocation limit')
        result = PyList_New(total)
        for part in range(len(chunks)):
            values = chunks[part]
            for i in range(values.shape[0]):
                index = values[i]
                if index >= self.size:
                    raise ValueError('Worker ID outside vocabulary')
                item = pool_item(<PyObject *>self.pool, index)
                incref(item)  # Only the new list slot steals this new reference.
                set_item(<PyObject *>result, position, item)
                position += 1
            values = None  # Release the export before dropping its owner.
            chunks[part] = None
        return result
