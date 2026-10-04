import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

# 实验二入口：计时从重型依赖导入之前开始，涵盖编码、训练、保存和验证。
# ruff: noqa: E402
import argparse
from contextlib import nullcontext
from datetime import UTC, datetime
import hashlib
from itertools import islice
import json
import math
from pathlib import Path

import numpy as np
import torch

from bpe import BPE
from transformer_module import AdamW, TransformerLM, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping
from wandb_record import experiment

BUDGET_SECONDS = 21600
ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
RESUME_KEYS = (
    *MODEL_KEYS,
    "batch_size",
    "gradient_accumulation",
    "learning_rate",
    "min_learning_rate",
    "warmup_steps",
    "cosine_steps",
    "weight_decay",
    "max_grad_norm",
    "precision",
    "seed",
    "optimizer_betas",
    "optimizer_eps",
)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("已到达累计时间上限；本次运行不能作为有效评分结果")


def autocast_context(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def encode_file(bpe, source, destination, deadline):
    """逐块读取原始文本，使用 BPE 流式接口保持跨块边界，再写入磁盘数组。"""
    digest = hashlib.sha256()
    count = 0
    partial = destination.with_suffix(".partial")

    def chunks():
        # newline="" 保留原文件的 CR/LF，不擅自规范化空白或插入特殊标记。
        with Path(source).open("r", encoding="utf-8", newline="") as handle:
            while text := handle.read(1 << 20):
                check_deadline(deadline)
                digest.update(text.encode("utf-8"))
                yield text

    try:
        ids = iter(bpe.encode_iterable(chunks()))
        with partial.open("wb") as handle:
            while block := list(islice(ids, 262144)):
                check_deadline(deadline)
                np.asarray(block, dtype="<u4").tofile(handle)
                count += len(block)
        if count < 2:
            raise ValueError(f"{source} 至少需要两个 token")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return np.memmap(destination, mode="r", dtype="<u4"), digest.hexdigest()


def validation_batches(tokens, context_length, batch_size, device):
    """每个目标 token 恰好计分一次；尾部补位只用于组批，不计入损失。"""
    for first in range(0, len(tokens) - 1, context_length * batch_size):
        starts = range(first, min(first + context_length * batch_size, len(tokens) - 1), context_length)
        lengths = [min(context_length, len(tokens) - 1 - start) for start in starts]
        width = max(lengths)
        x = np.zeros((len(lengths), width), dtype=np.int64)
        y = np.zeros_like(x)
        for row, (start, length) in enumerate(zip(starts, lengths)):
            x[row, :length] = tokens[start : start + length]
            y[row, :length] = tokens[start + 1 : start + length + 1]
        mask = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
        yield torch.tensor(x, device=device), torch.tensor(y, device=device), mask


@torch.no_grad()
def evaluate(model, tokens, context_length, batch_size, device, precision="fp32", deadline=math.inf):
    was_training = model.training
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    try:
        for x, y, mask in validation_batches(tokens, context_length, batch_size, device):
            check_deadline(deadline)
            with autocast_context(device, precision):
                losses = cross_entropy(model(x), y, reduction="none")
            total.add_(losses.masked_select(mask).double().sum())
            # 数量由数据长度确定，避免逐批将 GPU 上的 mask 求和取回 CPU。
            count += min(x.numel(), len(tokens) - 1 - count)
        synchronize(device)
        check_deadline(deadline)
        return total.item(), count
    finally:
        model.train(was_training)


@torch.no_grad()
def estimate_validation_seconds(model, tokens, config):
    """预热后测量几个验证批次，为最终全量验证保留时间；此过程同样计入预算。"""
    device = config["device"]
    model.eval()
    batches = validation_batches(tokens, config["context_length"], config["eval_batch_size"], device)
    durations = []
    for i, (x, y, mask) in enumerate(batches):
        check_deadline(config["deadline_clock"])
        synchronize(device)
        started = time.perf_counter()
        with autocast_context(device, config["precision"]):
            losses = cross_entropy(model(x), y, reduction="none")
            losses.masked_select(mask).sum()
        synchronize(device)
        if i:
            durations.append(time.perf_counter() - started)
        if i >= 3:
            break
    batches_count = math.ceil((len(tokens) - 1) / (config["context_length"] * config["eval_batch_size"]))
    return max(durations, default=time.perf_counter() - started) * batches_count


def train_model(config, run):
    device = config["device"]
    deadline = config["deadline_clock"]
    out = Path(config.get("output_dir") or ROOT / "outputs" / run.id)
    out.mkdir(parents=True, exist_ok=True)
    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    if set(bpe.vocab) != set(range(len(bpe.vocab))):
        raise ValueError("BPE 词表 ID 必须从零连续编号")
    # 哈希覆盖词表、合并顺序、特殊 token 清单，恢复时必须完全一致。
    tokenizer_hash = hashlib.sha256()
    for suffix in (".vocab.json", ".merges.json", ".config.json"):
        tokenizer_hash.update(Path(str(config["bpe_prefix"]) + suffix).read_bytes())
    identity = {"tokenizer_sha256": tokenizer_hash.hexdigest()}
    run.config.update(
        {
            "vocab_size": len(bpe.vocab),
            "special_tokens": bpe.special_tokens,
            "special_token_ids": {s: bpe._ids[s.encode("utf-8")] for s in bpe.special_tokens},
            "validation_policy": "nonoverlapping_windows_all_targets_except_first_token",
            "inserted_special_tokens": False,
        }
    )
    # 正式路径始终编码完整原文，编码与续训的耗时都计入累计预算。
    train, identity["train_sha256"] = encode_file(bpe, config["train_file"], out / "train.tokens", deadline)
    valid, identity["valid_sha256"] = encode_file(bpe, config["valid_file"], out / "valid.tokens", deadline)
    run.config.update({"train_token_count": len(train), "valid_token_count": len(valid)})
    (out / "config.json").write_text(
        json.dumps({k: v for k, v in config.items() if k != "deadline_clock"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def emit(metrics):
        run.log(metrics)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")

    if len(train) <= config["context_length"]:
        raise ValueError("训练数据长度必须超过上下文长度")
    run.config.update(identity)
    (out / "data_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = TransformerLM(len(bpe.vocab), **{k: config[k] for k in MODEL_KEYS}, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["optimizer_betas"]),
        eps=config["optimizer_eps"],
    )
    step = 0
    if config.get("resume"):
        # 仅恢复自己保存且可信的本地检查点，其中包含 NumPy 随机数状态。
        saved = torch.load(config["resume"], map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("续训数据或 BPE 与原运行不一致")
        if saved["run_id"] != config["resume_from_run"]:
            raise ValueError("resume_from_run 与检查点来源不一致")
        if any(saved["config"].get(key) != config.get(key) for key in RESUME_KEYS):
            raise ValueError("续训模型、优化器、采样或调度配置与原运行不一致")
        receipt_path = Path(config["resume"]).parent / "timing.json"
        if not receipt_path.is_file():
            raise ValueError("缺少原运行 timing.json，无法核对累计耗时")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["run_id"] != saved["run_id"] or config["previous_elapsed_seconds"] < receipt["time/total_seconds"]:
            raise ValueError("此前累计耗时不能少于原运行的实际累计耗时")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["iteration"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.device(device).type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    estimate = estimate_validation_seconds(model, valid, config)
    # 配置的预留时间是下限；测得的全量验证估计再乘安全系数，并留出保存时间。
    reserve = max(config["reserve_seconds"], 2 * estimate + config["save_reserve_seconds"])
    if time.perf_counter() + reserve >= deadline:
        raise TimeoutError("剩余预算不足以安全完成保存和全量验证")
    model.train()
    log_nll = torch.zeros((), device=device, dtype=torch.float64)
    log_tokens = torch.zeros((), device=device, dtype=torch.int64)
    interval_steps = 0
    last_step_seconds = 0.0
    tokens_per_step = config["batch_size"] * config["context_length"] * config["gradient_accumulation"]
    target_steps = config["max_steps"]

    def log_interval():
        nonlocal interval_steps
        nll, count = log_nll.item(), log_tokens.item()
        if count:
            if not math.isfinite(nll) or nll < 0:
                raise ValueError("训练损失异常")
            emit(
                {
                    "train/step": step,
                    "train/ppl": math.exp(nll / count),
                    "train/tokens": step * tokens_per_step,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "time/elapsed_seconds": config["previous_elapsed_seconds"]
                    + time.perf_counter()
                    - PROGRAM_STARTED_CLOCK,
                }
            )
            log_nll.zero_()
            log_tokens.zero_()
        interval_steps = 0

    while target_steps is None or step < target_steps:
        if time.perf_counter() + reserve + 2 * last_step_seconds >= deadline:
            break
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lr = get_lr_cosine_schedule(
            step, config["learning_rate"], config["min_learning_rate"], config["warmup_steps"], config["cosine_steps"]
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(config["gradient_accumulation"]):
            x, y = get_batch(train, config["batch_size"], config["context_length"], device, rng)
            with autocast_context(device, config["precision"]):
                loss = cross_entropy(model(x), y)
            # 日志累计未缩放的总 NLL；反传均值除以累积次数，等价于大批次均值。
            log_nll.add_(loss.detach().double() * y.numel())
            log_tokens.add_(y.numel())
            (loss / config["gradient_accumulation"]).backward()
        gradient_clipping(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        step += 1
        interval_steps += 1
        synchronize(device)
        last_step_seconds = time.perf_counter() - started
        if step % 100 == 0:
            log_interval()
    if interval_steps:
        log_interval()
    check_deadline(deadline)
    # 只保存最后模型，不按验证成绩挑选最佳检查点；模型文件仅保留在本地。
    checkpoint = out / "last.pt"
    partial = out / "last.pt.partial"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": step,
            "config": {k: config.get(k) for k in RESUME_KEYS},
            "identity": identity,
            "run_id": run.id,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
        },
        partial,
    )
    partial.replace(checkpoint)
    nll, count = evaluate(
        model, valid, config["context_length"], config["eval_batch_size"], device, config["precision"], deadline
    )
    synchronize(device)
    return {"val_nll_sum": nll, "val_token_count": count, "step": step, "training_tokens": step * tokens_per_step}


def parse_args():
    parser = argparse.ArgumentParser(description="手写 Transformer 训练与全量验证，累计预算最多六小时")
    parser.add_argument("--train-file", default=str(ROOT / "data/owt_train.txt"))
    parser.add_argument("--valid-file", default=str(ROOT / "data/owt_valid.txt"))
    parser.add_argument(
        "--bpe-prefix", default=str(ROOT / "outputs/830f59c92e15/tokenizer"), help="任务一保存的本人分词器路径前缀"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    for name, default in (
        ("context-length", 512),
        ("d-model", 512),
        ("num-layers", 4),
        ("num-heads", 8),
        ("d-ff", 1408),
        ("batch-size", 64),
        ("eval-batch-size", 16),
        ("gradient-accumulation", 1),
        ("warmup-steps", 200),
        ("cosine-steps", 20000),
        ("seed", 42),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (
        ("rope-theta", 10000),
        ("learning-rate", 3e-4),
        ("min-learning-rate", 3e-5),
        ("weight-decay", 0.1),
        ("max-grad-norm", 1.0),
        ("reserve-seconds", 120),
        ("save-reserve-seconds", 120),
        ("previous-elapsed-seconds", 0),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=BUDGET_SECONDS,
        help="累计墙钟预算，包含编码、训练、保存及验证；最多 21600 秒",
    )
    parser.add_argument("--max-steps", type=int, help="可选更新次数上限；默认按剩余预算停止")
    parser.add_argument("--resume", help="本地 last.pt 路径，需同时提供累计耗时及原 run ID")
    parser.add_argument("--resume-from-run")
    config = vars(parser.parse_args())
    budget = config["max_train_seconds"]
    if not math.isfinite(budget) or not 0 < budget <= BUDGET_SECONDS:
        parser.error("累计预算须在 0 到 21600 秒之间")
    config.update(optimizer_betas=[0.9, 0.999], optimizer_eps=1e-8)
    previous = config["previous_elapsed_seconds"]
    if not math.isfinite(previous) or not 0 <= previous < budget:
        parser.error("此前累计耗时无效或已经用完预算")
    if bool(config["resume"]) != bool(config["resume_from_run"]) or bool(config["resume"]) != (previous > 0):
        parser.error("续训须同时提供 resume、resume-from-run 和非零 previous-elapsed-seconds")
    positive = (
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "cosine_steps",
        "max_grad_norm",
        "rope_theta",
    )
    if any(config[k] <= 0 for k in positive):
        parser.error("模型维度、批次大小和累积次数等参数必须为正")
    if config["max_steps"] is not None and config["max_steps"] < 0:
        parser.error("max-steps 不能为负")
    if any(
        not math.isfinite(config[k]) or config[k] < 0
        for k in ("reserve_seconds", "save_reserve_seconds", "learning_rate", "min_learning_rate", "weight_decay")
    ):
        parser.error("学习率、权重衰减和预留时间须为非负有限值")
    if config["precision"] == "bf16" and (
        torch.device(config["device"]).type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        parser.error("bf16 模式需要支持 BF16 的 CUDA 设备")
    get_lr_cosine_schedule(
        0, config["learning_rate"], config["min_learning_rate"], config["warmup_steps"], config["cosine_steps"]
    )
    return config


def main():
    config = parse_args()
    previous = config["previous_elapsed_seconds"]
    with experiment("transformer", config, root=ROOT) as run:
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        config["deadline_clock"] = PROGRAM_STARTED_CLOCK + config["max_train_seconds"] - previous
        out = ROOT / "outputs" / run.id
        out.mkdir(parents=True, exist_ok=True)
        try:
            result = train_model(config, run)
        finally:
            # 正常退出和 Python 异常均记录耗时；强制杀进程时不能保证执行。
            try:
                synchronize(config["device"])
            finally:
                seconds = time.perf_counter() - PROGRAM_STARTED_CLOCK
                total = previous + seconds
                timing = {
                    "time/program_started_utc": datetime.fromtimestamp(PROGRAM_STARTED_UNIX, UTC).isoformat(),
                    "time/experiment_ended_utc": datetime.now(UTC).isoformat(),
                    "time/previous_elapsed_seconds": previous,
                    "time/current_run_seconds": seconds,
                    "time/total_seconds": total,
                    "time/difference_from_6h_seconds": total - BUDGET_SECONDS,
                    "time/within_6h": total <= BUDGET_SECONDS,
                    "time/within_requested_budget": total <= config["max_train_seconds"],
                }
                run.summary.update(timing)
                (out / "timing.json").write_text(json.dumps({"run_id": run.id, **timing}, indent=2), encoding="utf-8")
                print(json.dumps(timing, ensure_ascii=False), flush=True)
        if result["val_token_count"] <= 0:
            raise ValueError("验证 token 数必须大于零")
        loss = result["val_nll_sum"] / result["val_token_count"]
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("验证平均 NLL 必须是非负有限值")
        ppl = math.exp(loss)
        if not math.isfinite(ppl):
            raise ValueError("验证 PPL 非有限值")
        final = {
            "run_id": run.id,
            "val/ppl": ppl,
            "val/token_count": result["val_token_count"],
            "train/tokens": result["training_tokens"],
            "train/final_step": result["step"],
            "train/total_seconds": total,
        }
        run.summary.update(final)
        (out / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(json.dumps(final, ensure_ascii=False), flush=True)
        if total > config["max_train_seconds"]:
            raise TimeoutError("累计耗时超过本次设定的预算")


if __name__ == "__main__":
    main()
