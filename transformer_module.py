"""手写 Transformer、优化器与训练基础工具；仅依赖 PyTorch 基础张量运算。"""

import math

import numpy as np
import torch
from torch import nn


def linear(x, weight):
    return x @ weight.transpose(-1, -2)


class Linear(nn.Module):
    """无偏置线性层，权重按截断正态分布初始化。"""

    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, ids):
        return self.weight[ids]


def _accumulate_dtype(x):
    # 半精度输入升为单精度，双精度输入保持原精度。
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        values = _accumulate_dtype(x)
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


def silu(x):
    # 分别处理正负区间，避免 exp 在大幅度负输入处溢出。
    positive = torch.where(x >= 0, x, 0)
    negative = torch.where(x < 0, x, 0).exp()
    return torch.where(x >= 0, x / (1 + (-positive).exp()), x * negative / (1 + negative))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, **kwargs)
        self.w2 = Linear(d_ff, d_model, **kwargs)
        self.w3 = Linear(d_model, d_ff, **kwargs)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    """相邻的偶数、奇数维构成一对，按位置旋转 Q 和 K。"""

    def __init__(self, d_k, theta, max_seq_len, device=None):
        super().__init__()
        if d_k % 2 or theta <= 0 or max_seq_len <= 0:
            raise ValueError("RoPE 要求偶数维度、正 theta 和正上下文长度")
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = torch.arange(max_seq_len, device=device)[:, None] * frequencies
        # 缓存可以从配置重建，无须写入模型检查点。
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions=None):
        if positions is None:
            positions = torch.arange(x.shape[-2], device=x.device)
        cos, sin = self.cos[positions], self.sin[positions]
        while cos.ndim < x.ndim:
            cos, sin = cos.unsqueeze(-3), sin.unsqueeze(-3)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def softmax(x, dim=-1):
    values = _accumulate_dtype(x)
    shifted = values - values.amax(dim=dim, keepdim=True)
    exps = shifted.exp()
    return (exps / exps.sum(dim=dim, keepdim=True)).to(x.dtype)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, -1) @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=10000, **kwargs):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("模型维度必须能被注意力头数整除")
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, **kwargs)
        self.k_proj = Linear(d_model, d_model, **kwargs)
        self.v_proj = Linear(d_model, d_model, **kwargs)
        self.output_proj = Linear(d_model, d_model, **kwargs)
        self.rope = (
            None if max_seq_len is None else RoPE(d_model // num_heads, theta, max_seq_len, device=kwargs.get("device"))
        )

    def forward(self, x, token_positions=None):
        def split(projection):
            return projection(x).unflatten(-1, (self.num_heads, -1)).transpose(-3, -2)

        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        if self.rope is not None:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        length = x.shape[-2]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        output = scaled_dot_product_attention(q, k, v, mask)
        return self.output_proj(output.transpose(-3, -2).flatten(-2))


class TransformerBlock(nn.Module):
    """每个子层先归一化，再计算注意力或前馈网络，最后加残差。"""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, **kwargs):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, **kwargs)
        self.ln1 = RMSNorm(d_model, **kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **kwargs)
        self.ln2 = RMSNorm(d_model, **kwargs)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000, **kwargs):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, **kwargs)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **kwargs)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, **kwargs)
        self.lm_head = Linear(d_model, vocab_size, **kwargs)

    def forward(self, ids):
        if ids.shape[-1] > self.context_length:
            raise ValueError("输入超过模型上下文长度")
        x = self.token_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))


def cross_entropy(logits, targets, reduction="mean"):
    """通过减去最大值计算稳定的 log-sum-exp，避免先求概率再取对数。"""
    values = _accumulate_dtype(logits)
    shifted = values - values.amax(-1, keepdim=True)
    losses = shifted.exp().sum(-1).log() - shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction != "mean":
        raise ValueError("未知损失归约方式")
    return losses.mean()


@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    if max_l2_norm <= 0:
        raise ValueError("梯度范数上限必须为正")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    norm = torch.stack([_accumulate_dtype(g).square().sum() for g in grads]).sum().sqrt()
    scale = (max_l2_norm / (norm + 1e-6)).clamp(max=1)
    for grad in grads:
        grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    """手写一、二阶矩、偏差校正和解耦权重衰减。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps <= 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("AdamW 超参数不合法")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度")
                grad = _accumulate_dtype(p.grad)
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(grad), v=torch.zeros_like(grad))
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                denominator = (v / (1 - b2**t)).sqrt().add_(group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_((m / (1 - b1**t) / denominator).to(p.dtype), alpha=-group["lr"])
        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if not 0 <= warmup_iters < cosine_cycle_iters or not 0 <= min_learning_rate <= max_learning_rate:
        raise ValueError("学习率调度参数不合法")
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_learning_rate - min_learning_rate)


def get_batch(dataset, batch_size, context_length, device, rng=None):
    if dataset.ndim != 1 or len(dataset) <= context_length or batch_size <= 0 or context_length <= 0:
        raise ValueError("数据长度必须大于上下文长度，批大小与上下文长度必须为正")
    starts = (
        np.random.randint(0, len(dataset) - context_length, size=batch_size)
        if rng is None
        else rng.integers(0, len(dataset) - context_length, size=batch_size)
    )
    indices = starts[:, None] + np.arange(context_length + 1)
    batch = torch.tensor(np.asarray(dataset[indices], dtype=np.int64), device=device)
    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(model, optimizer, iteration, out):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration), out)


def load_checkpoint(src, model, optimizer):
    state = torch.load(src, map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
