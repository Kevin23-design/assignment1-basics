import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

import os

# 在 NumPy/PyTorch 导入前限制底层计算线程；保留环境中更小的配额。
CPU_THREADS = min(16, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
if __name__ == "__main__":
    # cuda:0 指向唯一可见 GPU；用户可在启动时指定物理编号或 UUID。
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:16]))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(min(CPU_THREADS, max(1, int(os.environ.get(name, CPU_THREADS)))))

# 实验二入口：计时从重型依赖导入之前开始，涵盖编码、训练、保存和验证。
# ruff: noqa: E402
from contextlib import nullcontext
from datetime import UTC, datetime
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

import numpy as np
import torch

from bpe import BPE
from parallel_bpe_encoding import THREAD_VARS, affinity
from transformer_module import AdamW, TransformerLM, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping
from wandb_record import experiment

BUDGET_SECONDS = 21600
ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
# 只在这里填写路径和训练配置；运行命令不需要额外参数。
CONFIG = {
    "train_file": "/data/assignment1-data/owt_train.txt",
    "valid_file": "/data/assignment1-data/owt_valid.txt",
    "bpe_prefix": str(ROOT / "outputs/e9275a3b4a7a/tokenizer"),
    "bpe_encode_workers": 16,
    "bpe_worker_cache_capacity": 1_048_576,
    "bpe_cache_policy": "clear_all",
    "bpe_split_special_token": "<|endoftext|>",
    "device": "cuda:0",  # 用 CUDA_VISIBLE_DEVICES 在启动命令中选择 GPU。
    "precision": "bf16",
    "num_layers": 4,
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 1408,
    "context_length": 512,
    "rope_theta": 10000,
    "batch_size": 64,
    "eval_batch_size": 16,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 500,
    "cosine_steps": 65000,
    "optimizer_betas": (0.9, 0.999),
    "optimizer_eps": 1e-8,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "max_train_seconds": 21600,
    # 历史保存+完整验证约 174 秒；下一版基线固定预留 10 分钟。
    "reserve_seconds": 600,
    "resume": None,
    "resume_from_run": None,
    "previous_elapsed_seconds": 0.0,
}


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
    count = 0
    partial = destination.with_suffix(".partial")

    def chunks():
        # newline="" 保留原文件的 CR/LF，不擅自规范化空白或插入特殊标记。
        with Path(source).open("r", encoding="utf-8", newline="") as handle:
            while text := handle.read(1 << 20):
                check_deadline(deadline)
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
    return np.memmap(destination, mode="r", dtype="<u4")


def encoding_cpus(workers):
    if not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("BPE worker 数须在 1..16 范围内")
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("正式多进程编码需要 Linux CPU affinity")
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    if len(cpus) != workers:
        raise ValueError(f"需要 {workers} 个 CPU，当前配额只有 {len(cpus)} 个，不自动降配")
    return cpus


def encode_file_parallel(config, source, destination, deadline):
    """重新编码原文；无 GPU 的协调进程退出后才返回磁盘 token 数组。"""
    check_deadline(deadline)
    workers = config["bpe_encode_workers"]
    cpus = encoding_cpus(workers)
    affinity(cpus, include_descendants=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 不能复用先前编码产物；与原 encode_file 一样重新写入完整数组。
    destination.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_VARS})
    environment["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory(prefix="bpe-job-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        job = {"source": str(source), "tokenizer_prefix": config["bpe_prefix"],
               "destination": str(destination), "workers": workers,
               "capacity": config["bpe_worker_cache_capacity"],
               "separator": config["bpe_split_special_token"], "cpus": cpus, "deadline": deadline}
        (folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
        command = [sys.executable, str(ROOT / "parallel_bpe_encoding.py"),
                   "--job", str(folder / "job.json"), "--result", str(folder / "result.json")]
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                check_deadline(deadline)
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"BPE 编码子进程失败，退出码 {process.returncode}")
            check_deadline(deadline)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            if result["status"] != "ok":
                raise ValueError(f"无法安全分为 {workers} 片：{result.get('reason')}")
            if result["token_count"] < 2 or destination.stat().st_size != result["token_count"] * 4:
                raise ValueError("编码结果不足两个 token 或 uint32 文件大小错误")
        except BaseException:
            # 整个子进程组包含协调进程及其 spawn worker；异常不得留下后台编码。
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            destination.unlink(missing_ok=True)
            Path(str(destination) + ".partial").unlink(missing_ok=True)
            shutil.rmtree(Path(str(destination) + ".parts"), ignore_errors=True)
            raise
    return np.memmap(destination, mode="r", dtype="<u4")


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
    val_nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    val_token_count = 0
    try:
        for x, y, mask in validation_batches(tokens, context_length, batch_size, device):
            check_deadline(deadline)
            with autocast_context(device, precision):
                losses = cross_entropy(model(x), y, reduction="none")
            val_nll_sum.add_(losses.masked_select(mask).double().sum())
            # 数量由数据长度确定，避免逐批将 GPU 上的 mask 求和取回 CPU。
            val_token_count += min(x.numel(), len(tokens) - 1 - val_token_count)
        synchronize(device)
        check_deadline(deadline)
        return val_nll_sum.item(), val_token_count
    finally:
        model.train(was_training)


def train_model(config, run):
    device = config["device"]
    deadline = config["deadline_clock"]
    out = Path(config.get("output_dir") or ROOT / "outputs" / run.id)
    out.mkdir(parents=True, exist_ok=True)
    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    if set(bpe.vocab) != set(range(len(bpe.vocab))):
        raise ValueError("BPE 词表 ID 必须从零连续编号")
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
    encoding_started = time.perf_counter()
    if config.get("bpe_encode_workers", 1) > 1:
        run.config.update({"bpe_cpu_affinity": encoding_cpus(config["bpe_encode_workers"]),
                           "bpe_worker_threads": 1, "bpe_worker_gpu_visible": False})
        train = encode_file_parallel(config, config["train_file"], out / "train.tokens", deadline)
        valid = encode_file_parallel(config, config["valid_file"], out / "valid.tokens", deadline)
    else:
        train = encode_file(bpe, config["train_file"], out / "train.tokens", deadline)
        valid = encode_file(bpe, config["valid_file"], out / "valid.tokens", deadline)
    run.summary["time/encode_seconds"] = time.perf_counter() - encoding_started
    run.config.update({"train_token_count": len(train), "valid_token_count": len(valid)})
    if len(train) <= config["context_length"]:
        raise ValueError("训练数据长度必须超过上下文长度")
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
    global_step = 0
    if config.get("resume"):
        # 仅恢复自己保存且可信的本地检查点，其中包含 NumPy 随机数状态。
        saved = torch.load(config["resume"], map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        global_step = saved["iteration"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.device(device).type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    reserve = config["reserve_seconds"]
    if time.perf_counter() + reserve >= deadline:
        raise TimeoutError("剩余预算不足以安全完成保存和全量验证")
    model.train()
    interval_nll = None
    interval_tokens = None
    interval_steps = 0
    interval_step_seconds = 0.0
    total_step_seconds = 0.0
    starting_step = global_step
    last_step_seconds = 0.0

    def log_interval():
        nonlocal interval_steps, interval_step_seconds
        logged_nll, logged_tokens = interval_nll.item(), interval_tokens.item()
        if logged_tokens > 0:
            if not math.isfinite(logged_nll) or logged_nll < 0:
                raise ValueError("训练损失异常")
            run.log(
                {
                    "train/step": int(global_step),
                    "train/target_tokens": global_step * config["batch_size"] * config["context_length"],
                    "train/loss": logged_nll / logged_tokens,
                    "train/ppl": math.exp(logged_nll / logged_tokens),
                    "train/lr": lr,  # 最后一次已完成更新使用的 LR，对应 global_step - 1。
                    "train/mean_step_seconds": interval_step_seconds / interval_steps,
                    "time/elapsed_seconds": config["previous_elapsed_seconds"]
                    + time.perf_counter()
                    - PROGRAM_STARTED_CLOCK,
                }
            )
        interval_nll.zero_()
        interval_tokens.zero_()
        interval_steps = 0
        interval_step_seconds = 0.0

    training_started = time.perf_counter()
    while True:
        if time.perf_counter() + reserve + 2 * last_step_seconds >= deadline:
            break
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lr = get_lr_cosine_schedule(
            global_step,
            config["learning_rate"],
            config["min_learning_rate"],
            config["warmup_steps"],
            config["cosine_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train, config["batch_size"], config["context_length"], device, rng)
        with autocast_context(device, config["precision"]):
            loss = cross_entropy(model(x), y)
        # 在设备上累计区间总 NLL，仅每 100 次更新取回日志数值。
        num_tokens = y.numel()
        loss_sum = loss.detach().to(dtype=torch.float64) * num_tokens
        if interval_nll is None:
            interval_nll = torch.zeros((), device=loss_sum.device, dtype=torch.float64)
            interval_tokens = torch.zeros((), device=loss_sum.device, dtype=torch.int64)
        interval_nll.add_(loss_sum.detach().to(dtype=torch.float64))
        interval_tokens.add_(torch.as_tensor(num_tokens, device=loss_sum.device, dtype=torch.int64).detach())
        loss.backward()
        gradient_clipping(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        global_step += 1
        interval_steps += 1
        synchronize(device)
        last_step_seconds = time.perf_counter() - started
        interval_step_seconds += last_step_seconds
        total_step_seconds += last_step_seconds
        if global_step % 100 == 0:
            log_interval()
    if interval_steps:
        log_interval()
    training_seconds = time.perf_counter() - training_started
    completed_steps = global_step - starting_step
    run.summary.update({
        "time/train_seconds": training_seconds,
        "train/completed_steps_this_run": completed_steps,
        "train/target_tokens": global_step * config["batch_size"] * config["context_length"],
        "train/target_tokens_this_run": completed_steps * config["batch_size"] * config["context_length"],
        "train/tokens_per_second": completed_steps * config["batch_size"] * config["context_length"]
        / training_seconds if training_seconds > 0 else 0.0,
        "train/steps_per_second": completed_steps / training_seconds if training_seconds > 0 else 0.0,
        "train/mean_step_seconds": total_step_seconds / completed_steps if completed_steps else 0.0,
    })
    check_deadline(deadline)
    # 只保存最后模型，不按验证成绩挑选最佳检查点；模型文件仅保留在本地。
    checkpoint = out / "last.pt"
    partial = out / "last.pt.partial"
    saving_started = time.perf_counter()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": global_step,
            "run_id": run.id,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
        },
        partial,
    )
    partial.replace(checkpoint)
    run.summary["time/save_seconds"] = time.perf_counter() - saving_started
    validation_started = time.perf_counter()
    val_nll_sum, val_token_count = evaluate(
        model, valid, config["context_length"], config["eval_batch_size"], device, config["precision"], deadline
    )
    synchronize(device)
    run.summary["time/validation_seconds"] = time.perf_counter() - validation_started
    if config.get("expected_validation_targets") is not None and val_token_count != config["expected_validation_targets"]:
        raise ValueError("完整验证目标数与实验要求不一致")
    return {"val_nll_sum": float(val_nll_sum), "val_token_count": int(val_token_count), "step": int(global_step)}


def main():
    config = CONFIG.copy()
    workers = config.get("bpe_encode_workers", 1)
    if not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("BPE worker 数须在 1..16 范围内")
    if config.get("bpe_encode_workers", 1) > 1:
        affinity(encoding_cpus(config["bpe_encode_workers"]), include_descendants=True)
    # ~ 使用实际登录用户的 home；相对路径以项目目录为基准。
    for key in ("train_file", "valid_file", "bpe_prefix", "resume", "output_dir"):
        if config.get(key):
            path = Path(config[key]).expanduser()
            config[key] = str(path if path.is_absolute() else ROOT / path)
    budget = config["max_train_seconds"]
    previous = config["previous_elapsed_seconds"]
    if not math.isfinite(budget) or not 0 < budget <= BUDGET_SECONDS:
        raise ValueError("累计预算须大于零且不超过六小时")
    if not math.isfinite(previous) or not 0 <= previous < budget:
        raise ValueError("此前累计耗时无效或已经用完预算")
    if bool(config["resume"]) != bool(config["resume_from_run"]) or bool(config["resume"]) != (previous > 0):
        raise ValueError("续训须填写检查点、原 run ID 和此前累计耗时")
    if not 0 < config["reserve_seconds"] < budget - previous:
        raise ValueError("须为保存和完整验证预留时间，且预留不能耗尽剩余预算")
    if config["precision"] == "bf16" and (
        torch.device(config["device"]).type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("BF16 需要支持该格式的 CUDA 设备")
    with experiment("transformer", config) as run:
        run.define_metric("train/step")
        run.define_metric("train/target_tokens", step_metric="train/step")
        for metric in ("train/ppl", "train/loss", "train/lr", "train/mean_step_seconds"):
            run.define_metric(metric, step_metric="train/step")
        config["deadline_clock"] = PROGRAM_STARTED_CLOCK + config["max_train_seconds"] - previous
        try:
            result = train_model(config, run)
        finally:
            # 正常退出和 Python 异常均记录耗时；强制杀进程时不能保证执行。
            try:
                synchronize(config["device"])
            finally:
                if torch.device(config["device"]).type == "cuda":
                    run.summary.update({
                        "gpu/peak_allocated_bytes": torch.cuda.max_memory_allocated(config["device"]),
                        "gpu/peak_reserved_bytes": torch.cuda.max_memory_reserved(config["device"]),
                    })
                ended_unix = time.time()
                current_seconds = time.perf_counter() - PROGRAM_STARTED_CLOCK
                total_seconds = previous + current_seconds
                timing = {
                    "time/program_started_utc": datetime.fromtimestamp(PROGRAM_STARTED_UNIX, UTC).isoformat(),
                    "time/experiment_ended_utc": datetime.fromtimestamp(ended_unix, UTC).isoformat(),
                    "time/previous_elapsed_seconds": previous,
                    "time/current_run_seconds": current_seconds,
                    "time/total_seconds": total_seconds,
                    "time/difference_from_6h_seconds": total_seconds - BUDGET_SECONDS,
                    "time/within_6h": total_seconds <= BUDGET_SECONDS,
                    "time/within_requested_budget": total_seconds <= config["max_train_seconds"],
                }
                run.summary.update(timing)
                print(json.dumps(timing, ensure_ascii=False), flush=True)
        if result["val_token_count"] <= 0:
            raise ValueError("验证 token 数必须大于零")
        val_loss = result["val_nll_sum"] / result["val_token_count"]
        if not math.isfinite(val_loss) or val_loss < 0:
            raise ValueError("验证平均 NLL 必须是非负有限值")
        final_ppl = math.exp(val_loss)
        if not math.isfinite(final_ppl):
            raise ValueError("验证 PPL 非有限值")
        final = {
            "run_id": run.id,
            "val/ppl": final_ppl,
            "val/loss": val_loss,
            "val/token_count": result["val_token_count"],
            "train/final_step": result["step"],
            "train/total_seconds": total_seconds,
        }
        run.summary.update(final)
        print(json.dumps(final, ensure_ascii=False), flush=True)
        if total_seconds > config["max_train_seconds"]:
            raise TimeoutError("累计耗时超过本次设定的预算")


if __name__ == "__main__":
    torch.set_num_threads(min(CPU_THREADS, int(os.environ["OMP_NUM_THREADS"])))
    torch.set_num_interop_threads(1)
    main()
