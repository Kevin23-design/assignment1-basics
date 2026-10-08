"""正式入口回归：完整编码、验证目标计数、保存恢复及累计计时。"""

import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from bpe import BPE
from transformer_main import encode_file, evaluate, train_model, validation_batches
from transformer_module import TransformerLM, cross_entropy


@pytest.mark.parametrize("length", [2, 8, 9, 10, 24, 25, 26])
def test_validation_preserves_state_and_counts_tail(length):
    model = TransformerLM(32, 8, 16, 1, 2, 32)
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.get_rng_state().clone()
    tokens = np.arange(length) % 32
    nll, count = evaluate(model, tokens, 8, 3, "cpu")
    assert model.training and torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    expected, targets = 0.0, []
    with torch.no_grad():
        for x, y, mask in validation_batches(tokens, 8, 1, "cpu"):
            expected += cross_entropy(model(x), y, "none")[mask].double().sum().item()
            targets.extend(y[mask].tolist())
    assert count == length - 1 and targets == tokens[1:].tolist()
    assert nll == pytest.approx(expected, rel=1e-6)
    model.eval()
    evaluate(model, tokens, 8, 3, "cpu")
    assert not model.training
    with pytest.raises(TimeoutError):
        model.train()
        evaluate(model, tokens, 8, 3, "cpu", deadline=0)
    assert model.training


@pytest.fixture
def config(tmp_path):
    tokenizer = BPE({i: bytes([i]) for i in range(256)}, [], ["<|endoftext|>"])
    prefix = tmp_path / "tokenizer"
    tokenizer.save(str(prefix))
    train, valid = tmp_path / "train.txt", tmp_path / "valid.txt"
    train.write_bytes(("中文 abc\r\n<|endoftext|>文档\n" * 20).encode())
    valid.write_bytes(("验证🙂 xy\r\n<|endoftext|>" * 3).encode())
    return dict(
        bpe_prefix=str(prefix),
        train_file=str(train),
        valid_file=str(valid),
        device="cpu",
        precision="fp32",
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10000,
        seed=42,
        learning_rate=0.001,
        min_learning_rate=0.0001,
        weight_decay=0.1,
        optimizer_betas=[0.9, 0.999],
        optimizer_eps=1e-8,
        warmup_steps=0,
        cosine_steps=200,
        batch_size=2,
        eval_batch_size=3,
        reserve_seconds=0,
        previous_elapsed_seconds=0,
        max_grad_norm=1.0,
        deadline_clock=time.perf_counter() + 300,
    )


def test_complete_encoding(config, tmp_path):
    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    from pathlib import Path

    for key in ("train_file", "valid_file"):
        ids = encode_file(bpe, config[key], tmp_path / (key + ".tokens"), time.perf_counter() + 30)
        raw = Path(config[key]).read_bytes()
        assert ids.tolist() == bpe.encode(raw.decode())
        assert bpe.decode(ids.tolist()).encode() == raw
    with pytest.raises(TimeoutError):
        encode_file(bpe, config["train_file"], tmp_path / "expired.tokens", 0)
    assert not (tmp_path / "expired.partial").exists()


def test_training_resume_and_full_validation(config, tmp_path, monkeypatch):
    import transformer_main as main

    def run(name, updates, **changes):
        logs, finished = [], []
        original_step = main.AdamW.step

        def step(optimizer):
            result = original_step(optimizer)
            finished.append(True)
            return result

        # 用可控时间到达固定收尾预留区，避免为测试保留额外训练分支。
        with monkeypatch.context() as patch:
            patch.setattr(main.time, "perf_counter", lambda: float(len(finished)))
            patch.setattr(main.AdamW, "step", step)
            result = train_model(
                {
                    **config,
                    "output_dir": str(tmp_path / name),
                    "deadline_clock": updates + 3,
                    "reserve_seconds": 1,
                    **changes,
                },
                SimpleNamespace(id=name, config={}, log=logs.append),
            )
        return result, logs

    full, logs = run("full", 3)
    assert full["step"] == 3 and [m["train/step"] for m in logs] == [3]
    run("part", 1)
    resumed, _ = run(
        "resumed", 2, resume=str(tmp_path / "part/last.pt"), resume_from_run="part", previous_elapsed_seconds=1.0
    )
    assert resumed == full
    saved = torch.load(tmp_path / "full/last.pt", weights_only=False)
    other = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert all(torch.equal(saved["model"][k], v) for k, v in other["model"].items())
    assert saved["iteration"] == 3
    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    from pathlib import Path

    tokens = bpe.encode(Path(config["valid_file"]).read_bytes().decode())
    assert full["val_token_count"] == len(tokens) - 1
    _, logs = run("logging", 101)
    assert [m["train/step"] for m in logs] == [100, 101]
    assert not list((tmp_path / "full").glob("*.json*"))


def test_main_records_course_metrics(config, tmp_path, monkeypatch):
    from contextlib import contextmanager
    import transformer_main as main

    run = SimpleNamespace(id="test", summary={}, define_metric=lambda *a, **kw: None)

    @contextmanager
    def experiment(task, config):
        assert task == "transformer"
        yield run

    cfg = {**config, "resume": None, "resume_from_run": None, "max_train_seconds": 100, "reserve_seconds": 20}
    monkeypatch.setattr(main, "CONFIG", cfg)
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "experiment", experiment)
    monkeypatch.setattr(main, "PROGRAM_STARTED_CLOCK", 0)
    monkeypatch.setattr(main.time, "perf_counter", lambda: 10.0)

    def train(config, run):
        assert config["deadline_clock"] == 100
        return dict(val_nll_sum=20.0, val_token_count=10, step=3)

    monkeypatch.setattr(main, "train_model", train)
    main.main()
    assert run.summary["val/ppl"] == pytest.approx(np.exp(2))
    assert run.summary["time/total_seconds"] == 10
    assert run.summary["train/final_step"] == 3
    cfg["previous_elapsed_seconds"] = 1
    with pytest.raises(ValueError, match="续训"):
        main.main()
