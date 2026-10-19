"""General Python dictionary-lookup fallback; join bytes before UTF-8 decoding."""
from operator import itemgetter
BLOCK = 131072

def raw_chunks(vocab, ids):
    for start in range(0, len(ids), BLOCK):
        block = ids[start:start + BLOCK]
        values = itemgetter(*block)(vocab)
        yield b''.join((values,) if len(block) == 1 else values)


def serial(vocab, ids):
    return b''.join(raw_chunks(vocab, ids)).decode('utf-8', errors='replace')
