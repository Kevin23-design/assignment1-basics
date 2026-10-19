"""Small production-Q3 checks. Explicitly build first; no OWT benchmark or training."""
from array import array
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import bpe as implementation
from bpe import BPE
import bpe_memory_runtime as runtime
import bpe_native_deployment as deployment

ROOT = Path(__file__).resolve().parents[1]


def sample():
    return 'ab中文🙂\r\n \t<x>long尾<x><|endoftext|>' * 650 + '终🙂'


def tokenizer():
    return BPE({**{i: bytes([i]) for i in range(256)}, 256: b'ab'}, [(b'a', b'b')],
               ['<x>', '<x>long', '<|endoftext|>', '<absent>'])


def python_reference(text):
    model = tokenizer()
    model._encode_uncached = model._python_encode_uncached
    return model._encode_serial(text)


@pytest.fixture
def native():
    if not (ROOT / 'bpe_native/deployment.json').is_file():
        pytest.skip('Explicitly run build_bpe_native.py to test production native integration')
    deployment.check()
    yield


def test_public_interface_cache_save_load_and_standalone_decode(native, tmp_path):
    model, text = tokenizer(), sample()
    assert model._use_native
    ids = model.encode(text)
    assert type(ids) is list and ids == python_reference(text)
    assert model.decode(ids) == text
    assert model._native_decode_metrics['decoder_case'] == 'N2'
    assert model.decode([0xff]) == '\ufffd'
    for invalid in (-1, 999999, 10**100):
        with pytest.raises(KeyError):
            model.decode([invalid])
    model.vocab[256] = b'changed'
    assert model.decode([256]) == 'changed'
    model = tokenizer()
    model.save(str(tmp_path / 'tokenizer'))
    loaded = BPE()
    loaded.load(str(tmp_path / 'tokenizer'))
    assert loaded.vocab == model.vocab and loaded.merges == model.merges
    assert loaded.special_tokens == model.special_tokens
    assert loaded.encode(text) == ids and loaded.decode(ids) == text


def test_lazy_stream_remains_python_2k_without_native_table(native):
    model, text = tokenizer(), sample()
    stream = model.encode_iterable(text[i:i+3] for i in range(0, len(text), 3))
    assert model._native_encoder_state is None and model._cache == {}
    ids = list(stream)
    assert ids == python_reference(text)
    assert len(model._cache) <= 2048 and model._native_encoder_state is None
    assert model._native_encode_metrics is None


def test_q3_public_path_two_clean_workers(native, monkeypatch):
    monkeypatch.setattr(implementation, '_PARALLEL_MIN_CHARACTERS', 1)
    monkeypatch.setattr(implementation, '_ENCODE_WORKERS', 2)
    monkeypatch.setattr(implementation, '_PARTITIONS_PER_WORKER', 128)
    def local_parallel(model, text, workers, multiplier):
        ranges = runtime._partition_ranges(model, text, workers * multiplier)
        assert len(ranges) == 256
        return runtime._launch(model, text, ranges, [], workers)
    monkeypatch.setattr(runtime, 'encode_if_parallel', local_parallel)
    model, text = tokenizer(), sample()
    ids = model.encode(text)
    assert ids == python_reference(text) and model.decode(ids) == text
    assert model._task1_assembly_metrics['method'] == 'Q3'
    assert model._task1_assembly_metrics['converted_ids'] == len(ids)
    assert model._task1_assembly_metrics['fragments'] == 256
    assert sum(x['tasks'] for x in model._task1_worker_metrics) == 256
    assert all(x['native_encoding']['native_calls'] > 0 and not x['native_encoding']['fallback_calls']
               for x in model._task1_worker_metrics)


def test_sparse_vocab_and_insufficient_boundaries_fall_back(native):
    model = BPE({0: b'a', 300: b'<x>'}, [], ['<x>'])
    assert runtime.encode_if_parallel(model, 'a<x>' * 1000, 16, 16) is None
    model = tokenizer()
    assert runtime._partition_ranges(model, 'ab<x>', 256) is None
    assert model.encode('ab<x>') == model._encode_serial('ab<x>')


def test_missing_deployment_strict_mode_and_python_save_load(tmp_path, monkeypatch):
    old_root = deployment.ROOT
    monkeypatch.setattr(deployment, 'ROOT', tmp_path)
    deployment.enabled.cache_clear()
    monkeypatch.setenv('BPE_REQUIRE_NATIVE', '1')
    try:
        with pytest.raises(RuntimeError, match='Compile native BPE first'):
            BPE()
        monkeypatch.delenv('BPE_REQUIRE_NATIVE')
        deployment.enabled.cache_clear()
        model = tokenizer()
        assert not model._use_native
        prefix = str(tmp_path / 'tokenizer')
        model.save(prefix)
        model.load(prefix)
        assert model._encode_uncached.__self__ is model
        assert model.decode(model.encode(sample())) == sample()
    finally:
        monkeypatch.setattr(deployment, 'ROOT', old_root)
        deployment.enabled.cache_clear()


def test_bad_first_deployment_check_is_not_cached(tmp_path, monkeypatch):
    (tmp_path / 'bpe_native').mkdir()
    path = tmp_path / 'bpe_native/deployment.json'
    path.write_text('{"schema": 999, "backend": "Q3"}')
    monkeypatch.setattr(deployment, 'ROOT', tmp_path)
    deployment.enabled.cache_clear()
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='Unknown BPE native deployment'):
                deployment.enabled()
        assert deployment.enabled.cache_info().currsize == 0
    finally:
        deployment.enabled.cache_clear()


def test_check_does_not_warm_extensions(native):
    code = '''import sys
import bpe_native_deployment as d
d.check()
from bpe import BPE
model=BPE()
assert model._use_native
assert not any(name in sys.modules for name in ('_task1_native','_task1_decode_access','_task1_parent_assembly'))
'''
    subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT, timeout=15)


def test_pool_actual_lengths_and_lifetime(native):
    from task1_parent_pipeline_runtime import load_parent
    pool = load_parent().Pool(32000)
    chunks = [array('I', [10000, 10000]), array('I', [2])]
    ids = pool.assemble_owned(chunks)
    assert ids == [10000, 10000, 2] and ids[0] is ids[1]
    assert chunks == [None, None]
    del pool
    gc.collect()
    assert ids == [10000, 10000, 2]


def test_course_entry_is_unmodified():
    # Exact template preservation was explicitly requested repeatedly.
    expected = '0145970b8cf675d3730d4393f397b4eaeb6ed0502b4214f819f50a1280e97b64'
    assert hashlib.sha256((ROOT / 'train_bpe.py').read_bytes()).hexdigest() == expected


def test_course_recorder_includes_project_native_sources_and_manifest(tmp_path, monkeypatch):
    import wandb_record as record
    root = tmp_path
    (root / 'bpe_native/native').mkdir(parents=True)
    for name, value in [('helper.pyx', b'def f():return 1\n'), ('helper.c', b'/* generated C */\n'),
                        ('helper.so', b'\x00\xff'), ('build.json', b'{"abi":"test"}')]:
        (root / 'bpe_native/native' / name).write_bytes(value)
    (root / 'outputs').mkdir()
    (root / 'outputs/hidden.py').write_text('hidden')
    artifacts = []
    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []
            artifacts.append(self)
        def add_file(self, path, name):
            self.files.append(name)
    class Run:
        id, summary = 'native-test', {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def log_artifact(self, value): pass
        def save(self, *args, **kwargs): pass
    run = Run()
    monkeypatch.setattr(record, 'wandb', SimpleNamespace(init=lambda **kw: run, Artifact=Artifact, Settings=lambda **kw: kw))
    for key in ('WANDB_ENTITY', 'STUDENT_ID'):
        monkeypatch.setenv(key, 'test')
    with record.experiment('bpe', {}, root=root):
        pass
    files = set(artifacts[0].files)
    assert {'bpe_native/native/helper.pyx', 'bpe_native/native/helper.c', 'bpe_native/native/helper.so',
            'bpe_native/native/build.json'} <= files
    assert 'outputs/hidden.py' not in files and run.summary['source_changed'] == []


def test_worker_failure_reclaims_children(native, monkeypatch):
    import task1_transport_runtime as transport
    processes = []
    launch = transport.subprocess.Popen
    def capture(*args, **kwargs):
        process = launch(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(transport.subprocess, 'Popen', capture)
    model = tokenizer()
    model.vocab[0] = object()
    ranges = runtime._partition_ranges(model, sample(), 256)
    with pytest.raises(RuntimeError):
        runtime._launch(model, sample(), ranges, [], 2)
    assert processes and all(p.poll() is not None for p in processes)


@pytest.mark.parametrize('signal_name', ['SIGINT', 'SIGTERM'])
def test_release_signal_cleanup(native, tmp_path, signal_name):
    import signal
    import time
    import psutil  # Test-only observer, never part of submitted BPE execution.
    ready = tmp_path / 'ready.json'
    code = f'''
import json,time
from pathlib import Path
from bpe import BPE
import bpe_memory_runtime as r
import task1_transport_runtime as t
model=BPE(special_tokens=['<x>'])
original=t.await_ready
def wait(*args):
 original(*args)
 Path({str(ready)!r}).write_text(json.dumps([p.pid for p in args[1]]))
 time.sleep(30)
t.await_ready=wait
text='ab<x>'*1000
r._launch(model,text,r._partition_ranges(model,text,256),[],2)
'''
    with (tmp_path / 'signal.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists()
            import json
            children = json.loads(ready.read_text())
            process.send_signal(getattr(signal, signal_name))
            process.wait(timeout=10)
            assert process.returncode != 0 and all(not psutil.pid_exists(p) for p in children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_unmodified_course_template_full_timing_and_200_token_table(native, tmp_path, monkeypatch):
    import builtins
    from contextlib import contextmanager
    import io
    import train_bpe as entry
    text = ('ab 中文🙂\r\n <|endoftext|> ' * 100) + '尾'
    source = tmp_path / 'small-train.txt'
    source.write_bytes(text.encode())
    actual_train, actual_encode = BPE.train, BPE.encode
    def train(self, path, size, special):
        assert path == '/data/assignment1-data/owt_train.txt' and size == 32000
        return actual_train(self, str(source), size, special)
    def encode(self, value):
        assert not self._cache and self._native_encoder_state is None
        return actual_encode(self, value)
    actual_open = builtins.open
    def read(path, *args, **kwargs):
        if str(path) == '/data/assignment1-data/owt_valid.txt':
            assert kwargs == {'encoding': 'utf-8', 'newline': ''}
            return io.StringIO(text)
        return actual_open(path, *args, **kwargs)
    class Run:
        id, summary, tables = 'course-test', {}, []
        def log(self, value):
            self.tables.append(value['bpe/first_200_tokens'])
    run = Run()
    @contextmanager
    def experiment(task, config):
        assert task == 'bpe' and config['special_tokens'] == ['<|endoftext|>']
        yield run
    stamps = iter([100., 101.5])
    monkeypatch.setattr(entry, 'time', SimpleNamespace(perf_counter=lambda: next(stamps)))
    monkeypatch.setattr(entry, 'experiment', experiment)
    monkeypatch.setattr(BPE, 'train', train)
    monkeypatch.setattr(BPE, 'encode', encode)
    monkeypatch.setattr(builtins, 'open', read)
    monkeypatch.chdir(tmp_path)
    entry.main()
    assert run.summary['bpe/valid_seconds'] == 1.5 and run.summary['bpe/roundtrip_passed']
    assert run.summary['bpe/valid_token_count'] >= 200
    assert len(run.tables[0].data) == 200 and all(row[-1] for row in run.tables[0].data)
    assert (tmp_path / 'outputs/course-test/tokenizer.config.json').is_file()
