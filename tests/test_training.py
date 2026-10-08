"""正式入口回归：完整编码、验证目标计数、保存恢复及累计计时。"""

import json
import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from bpe import BPE
from transformer_main import encode_file, evaluate, parse_args, train_model, validation_batches
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
        gradient_accumulation=2,
        reserve_seconds=0,
        save_reserve_seconds=0,
        previous_elapsed_seconds=0,
        max_grad_norm=1.0,
        max_steps=3,
        deadline_clock=time.perf_counter() + 300,
    )


def test_complete_encoding(config, tmp_path):
    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    from pathlib import Path

    for key in ("train_file", "valid_file"):
        ids, digest = encode_file(bpe, config[key], tmp_path / (key + ".tokens"), time.perf_counter() + 30)
        raw = Path(config[key]).read_bytes()
        assert ids.tolist() == bpe.encode(raw.decode())
        assert bpe.decode(ids.tolist()).encode() == raw
        import hashlib

        assert digest == hashlib.sha256(raw).hexdigest()
    with pytest.raises(TimeoutError):
        encode_file(bpe, config["train_file"], tmp_path / "expired.tokens", 0)
    assert not (tmp_path / "expired.partial").exists()


def test_training_resume_and_full_validation(config, tmp_path):
    def run(name, **changes):
        logs = []
        result = train_model(
            {**config, "output_dir": str(tmp_path / name), **changes},
            SimpleNamespace(id=name, config={}, log=logs.append),
        )
        return result, logs

    full, logs = run("full")
    assert full["step"] == 3 and full["training_tokens"] == 96
    assert [m["train/step"] for m in logs] == [3]
    assert all(not any(k.startswith("dev/") for k in m) for m in logs)
    run("part", max_steps=1)
    (tmp_path / "part/timing.json").write_text(json.dumps({"run_id": "part", "time/total_seconds": 1.0}))
    resumed, _ = run(
        "resumed", resume=str(tmp_path / "part/last.pt"), resume_from_run="part", previous_elapsed_seconds=1.0
    )
    assert resumed == full
    saved = torch.load(tmp_path / "full/last.pt", weights_only=False)
    other = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert saved["iteration"] == 3
    assert all(torch.equal(saved["model"][k], v) for k, v in other["model"].items())
    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    from pathlib import Path

    tokens = bpe.encode(Path(config["valid_file"]).read_bytes().decode())
    assert full["val_token_count"] == len(tokens) - 1
    with pytest.raises(ValueError, match="累计耗时"):
        run("bad_time", resume=str(tmp_path / "part/last.pt"), resume_from_run="part", previous_elapsed_seconds=0.5)
    with pytest.raises(ValueError, match="配置"):
        run(
            "bad_config",
            resume=str(tmp_path / "part/last.pt"),
            resume_from_run="part",
            previous_elapsed_seconds=1.0,
            learning_rate=0.002,
        )
    _, logs = run("logging", max_steps=101)
    assert [m["train/step"] for m in logs] == [100, 101]
    assert [m["train/tokens"] for m in logs] == [3200, 3232]


def test_budget_reserves_final_save_and_validation(config, tmp_path, monkeypatch):
    import transformer_main as main

    logs = []
    config = {
        **config,
        "output_dir": str(tmp_path / "budget"),
        "max_steps": None,
        "reserve_seconds": 10,
        "deadline_clock": 100.0,
    }
    # 第 100 次更新写日志后进入收尾预留区，必须自动结束训练并保存、验证。
    monkeypatch.setattr(main.time, "perf_counter", lambda: 95.0 if logs else 0.0)
    result = train_model(config, SimpleNamespace(id="budget", config={}, log=logs.append))
    assert result["step"] == 100 and (tmp_path / "budget/last.pt").is_file()


def test_formal_defaults_and_removed_development_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["transformer_main.py", "--device", "cpu", "--precision", "fp32"])
    cfg = parse_args()
    assert (cfg["num_layers"], cfg["d_model"], cfg["num_heads"], cfg["d_ff"], cfg["context_length"]) == (
        4,
        512,
        8,
        1408,
        512,
    )
    assert cfg["batch_size"] * cfg["gradient_accumulation"] * cfg["context_length"] == 32768
    assert cfg["optimizer_betas"] == [0.9, 0.999] and cfg["max_steps"] is None
    for args in (
        ["--development"],
        ["--dev-manifest", "full.json"],
        ["--train-max-chars", "100"],
        ["--max-train-seconds", "21601"],
        ["--resume", "last.pt"],
    ):
        monkeypatch.setattr("sys.argv", ["transformer_main.py", "--device", "cpu", "--precision", "fp32", *args])
        with pytest.raises(SystemExit):
            parse_args()
