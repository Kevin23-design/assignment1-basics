# BPE 与 Transformer 作业

正式源码包括 `bpe.py`、`train_bpe.py`、`transformer_module.py`、
`transformer_main.py`、`parallel_bpe_encoding.py`、`bpe_memory_runtime.py` 和课程提供的 `wandb_record.py`。
模型核心和BPE训练/合并规则保持不变；记录器增加原生源码、构建清单与二进制覆盖。

## 环境与运行

```sh
source /env/activate_a1.sh
wandb login
export WANDB_ENTITY="你的实际账号或团队"
export WANDB_PROJECT="lmfs-assignment1"
export STUDENT_ID="你的学号"
export WANDB_MODE="online"
export PYTHONIOENCODING="utf-8"
```

任务一正式入口继续保持课程 `train_bpe.py` 模板。Cython 使用依据为你在本次会话确认的课程明确许可；粘贴版允许列表本身未写明 Cython。

另：此前你提过禁止AI生成核心代码，本次优化含AI辅助实现；若该条仍适用，Cython许可不能替代代码来源许可。详见[BPE_Q3_REVIEW.md](BPE_Q3_REVIEW.md)。

现接入第 6 轮最快候选 **Q3**：16 workers / 256 安全分片 / 每 worker 1M 满后清空缓存；
先派发下一任务，再处理当前结果；按实际返回数组长度分配最终完整 `list[int]`，不按参考 token 数预分配。
解码使用 N2：在本次 `decode()` 内验证并建立 uint32 索引，拼接 bytes 后完整 UTF-8 `errors="replace"` 解码；
不使用 encode 时的原文或保留数组。公开接口、词表/merges/特殊 token JSON 格式不变。
默认惰性 `encode_iterable()` 保留原 Python 合并核心与 2K 缓存，避免为流式接口增加原生合并表的内存。

**结果范围须区分：**Q0/Q3 配对中位数 **6.127/5.676 秒**，降低 **7.36%**，5/5 胜出；
独立确认 **5.449 秒**，普通接口接入中位数 **5.701 秒**。接入中位数比例满足门槛，但仅 **3/5** 对满足单对105%限制，
未达到4/5要求，`integration.passed=false`；本轮未提供活跃 W&B 独立确认。Q3是当前最快候选，
**不是已经完整通过接入稳定性验收或已经取得的新正式成绩**。不放宽门槛、不改写历史报告，云端还需复核。

所有自编 Python/`.pyx` 源码放在项目目录。`build_bpe_native.py` 显式编译三个独立扩展到项目内
`bpe_native/`，包括输入源码副本、生成 C、ABI/编译器/参数/指纹清单及二进制；不依赖 `outputs` 中的探索快照。
该生成目录不提交 Git，但课程记录器明确收录其中源码、清单和二进制，不受 Git ignore 影响。
首次扩展导入、worker 原生表准备、所有进程启动/退出和最终列表组装仍在公开接口计时内。
不自动编译；源文件、ABI或编译产物变化会明确报错。构建检查不导入扩展或编码任何文本。

使用你已确认的个人解释器，**不修改公共环境或项目依赖文件**；不使用裸 `python` 或 `uv run`：

```bash
source /env/activate_a1.sh
bpe_python="/home/zhangziyang/miniforge3/envs/bpe-native/bin/python"

# 首次编译；已有且完全匹配则复用。源码变化后显式添加 --rebuild。
"$bpe_python" build_bpe_native.py
"$bpe_python" build_bpe_native.py --check

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
bpe_cpus="$("$bpe_python" -c 'import os; c=sorted(os.sched_getaffinity(0)); assert len(c)>=16, "需要至少16个允许CPU"; print(",".join(map(str,c[:16])))')"

# 在 W&B 服务启动前即固定16核，隐藏GPU；缺少原生构建时不允许回退冒充Q3。
taskset -c "$bpe_cpus" env CUDA_VISIBLE_DEVICES="" BPE_REQUIRE_NATIVE=1 \
  "$bpe_python" train_bpe.py
```

运行前仍需设置个人 `WANDB_ENTITY`、`WANDB_PROJECT`、`STUDENT_ID` 并登录个人账号。
正式入口从完整 `/data/assignment1-data/owt_train.txt` 训练、保存并重新加载，完整读取验证原文；
评分计时仅包含 `encode → 完整list[int] → decode → 原文比较`，200 token详情与上传在计时后。
BPE训练没有六小时时限；Transformer的六小时规则不改。最终以新run的 `bpe/valid_seconds` 作为成绩。
任务一正式来源凭证是这次完整训练的 W&B run，旧词表的测速记录不能替代其训练来源证明。

没有构建且未设 `BPE_REQUIRE_NATIVE=1` 时保留原 Python 兼容路径；`BPE_PYTHON_ONLY=1` 可显式使用它。
两者同时设置会报错。正常构建后默认使用Q3，正式命令显式要求原生，防止无意跑成较慢的兼容实现。
小文本、无足够安全切点、非Linux或非连续uint32 ID保留串行兼容语义；有足够切点而不足16核时不静默降配。
分片仅在实际匹配的完整特殊 token 后，较长标记优先，worker不导入W&B/Transformer，不使用fork或协调层。
异常或信号会回收全部本次worker，不返回部分ID。公共数据只读，不生成额外标记，不调用第三方Tokenizer。

本地只跑小文本；Linux affinity与官方1MB流式内存限制需在云端确认：

```bash
"$bpe_python" -m pytest tests/test_bpe_native_integration.py tests/test_tokenizer.py tests/test_train_bpe.py -q
```

任务二只修改 `transformer_main.py` 顶部的 `CONFIG` 字典，然后在项目根目录执行：

```sh
CUDA_VISIBLE_DEVICES=0 python transformer_main.py
```

不再使用命令行参数、开发清单、梯度累积或自动验证耗时估算。
配置中的原始训练和验证文件必须是完整 OWT 文件。分词器默认指向本人已训练的
`outputs/e9275a3b4a7a/tokenizer`；迁移时携带同前缀的 vocab、merges、config 三个 JSON 文件。

原文默认读取 `/data/assignment1-data/owt_train.txt` 和 `owt_valid.txt`。
公共环境只激活使用，不执行 `uv sync` 或修改依赖。

正式编码采用云端已确认的 **16 workers × 每 worker 1,048,576 项、满后清空**：
验证集三轮中位数 7.356 秒；完整训练集一次确认 183.402 秒，8 workers 为 336.431 秒。
两种训练集配置均通过完整 ID 对照和原文字节还原，16 workers 的产物 SHA-256 与参考相同。
这些是探索成绩，新的正式运行仍从入口计时，重新编码完整 train/valid，包含启动、分片、
读写、合并与清理时间，不加载探索产物跳过编码。

`CONFIG` 中的 `bpe_encode_workers=16`、`bpe_worker_cache_capacity=1048576` 和
`bpe_split_special_token="<|endoftext|>"` 控制编码。Linux 当前可用逻辑 CPU 少于配置数量
时直接报错，不自动降配。整个程序 CPU affinity 最多 16 个 CPU，worker 各绑定一个 CPU，
内部线程为 1 且看不到 GPU；协调进程和所有 worker 退出后才初始化模型。父进程保留一张
可见 GPU，`CUDA_VISIBLE_DEVICES` 可指定物理编号或 UUID，代码始终使用 `cuda:0`。
W&B 配置记录 worker 数、容量、策略与实际 CPU 集合，源码记录器同时包含辅助模块。

只能在分词器实际识别到的完整特殊 token 之后分片；不足所需非空分片时正式运行报错。
六小时截止检查覆盖切点扫描、worker 读写、等待和拼接，异常清理临时分片，不进入训练。
默认 BPE `encode_iterable()` 仍保持原有 2K 缓存；`bpe_encode_workers=1` 使用原单进程
路径，供调试与兼容旧配置。续训继续使用原 run 的分词器及模型配置。

模型配置为 4 层、512 维、8 头、FFN 1408、上下文 512、BF16、RoPE θ=10,000。
batch 64，每次更新 32,768 个目标 token；eval batch 16，seed 42。
AdamW β=(0.9,0.999)、ε=1e-8、权重衰减 0.1、梯度裁剪 1.0。
当前默认采用已测 J 配置：峰值学习率 **2e-3**、最低学习率 **3e-5**，warmup **500** 步，
余弦终点为全局 step **65,000**；终点后保持最低学习率，不按步数提前结束。
固定预留 **600 秒**收尾，累计预算仍为 **21,600 秒**，16-worker 编码配置保持不变。
本次 `CONFIG` 仅将峰值 LR 从 3e-4 更新为 2e-3，其余参数已与成功 J 运行匹配。

截至 2026-10-19，现有完整验证记录中采用 J 为下一次正式运行的默认设置。
同为 512 验证窗口的最近三组结果：

| 组别 | 峰值 LR | 完整验证 PPL | 最终 step | 训练目标 token |
|---|---:|---:|---:|---:|
| G | 6e-4 | 38.10523 | 54,670 | 1,791,426,560 |
| H | 1e-3 | 36.14949 | 54,940 | 1,800,273,920 |
| **J** | **2e-3** | **33.40074** | **77,282** | **2,532,376,576** |

[J 的 W&B 记录](https://wandb.ai/ziyang-zhang-east-china-normal-university/lmfs-assignment1/runs/785eef0bb430)
显示完整验证 loss **3.50857794**、目标数 **66,401,097**，累计 **21,174.21 秒**，
低于六小时，源码记录 `source_changed=[]`；启动器也确认源码未变且退出码为零。
编码 **192.12 秒**、训练 **20,783.21 秒**、保存 **0.77 秒**、完整验证 **173.60 秒**。
目标设备为 NVIDIA A100 80GB PCIe，实测训练吞吐约 **121,847 token/s**。
这些是历史实验成绩，新运行仍需从随机权重开始，重新编码完整原文并完成全量验证。

J 相比 H 的验证 PPL 降低 **7.60%**，但训练目标 token 也增加 **40.67%**；
单卡与多卡并发时的主机负载不同，不能将全部收益归因于学习率，也不声称单次比较具有
统计显著性。256 窗口的 C/D/E/F/I 不与本表混为相同验证口径排名。更改学习率或路径
直接修改 `CONFIG`；旧检查点续训必须恢复原 run 的调度与超参，不能套用新的默认值。

优先查看 W&B 的 `train/loss` 曲线；它是区间总 NLL / 总目标 token 数，`train/ppl`
为该 loss 的指数。每 100 次更新及最后不足 100 步的区间记录 `train/lr` 和
`train/mean_step_seconds`，横轴均为 `train/step`。LR 是最后一次已完成更新的值，
调度索引对应显示 step 减一；平均步耗时包括数据采样、前后向、更新与同步，不包含日志上传。
设备端累计方式保持不变，不新增逐步 `.item()`。

Summary 增加 `time/encode_seconds`（完整 train/valid 编码）、`time/train_seconds`
（本次训练循环，含区间日志）、`time/save_seconds`、`time/validation_seconds` 和
`val/loss`。`train/steps_per_second` 按本次完成的更新数 / 本次训练循环耗时计算；
Summary 的 `train/mean_step_seconds` 是本次所有已完成更新的平均耗时。分阶段耗时不覆盖
全部初始化操作，六小时预算仍以入口起算的累计墙钟耗时为准。所有新增记录操作也计入预算。

原始历史基线完整验证 PPL 为 49.16284；它与 J 的训练安排、实际训练量均不同。
J 的保存与完整验证约 174.37 秒，为继续保留 600 秒收尾预留提供实测依据。
新的正式运行仍须完整验证且累计不超过六小时，不能将历史 J 的成绩当作新 run 的成绩。

## 课程要求保留项

- 使用本人 BPE 编码完整原文，不额外插入标记；模型从随机权重开始。
- 编码后写入磁盘 token 数组，从整个训练数组随机抽取连续窗口。
- 模型各层、注意力、交叉熵、AdamW、调度和梯度裁剪均为手写实现。
- 每 100 次更新向 W&B 记录区间训练 loss、PPL、最后使用的 LR 和平均步耗时，尾部也记录。
- 仅保存最后模型及优化器、步数和随机状态到本地 `last.pt`，随后验证完整 OWT 验证集。
- 验证关闭梯度，按总目标 NLL / 总目标数计算 PPL，尾部计分，仅首 token 无预测目标。
- 从入口最顶部计时，导入、W&B、源码、编码、初始化、训练、保存和最终验证均计入累计六小时。
- 固定预留 600 秒收尾，不自动估算；运行期间仍检查截止时间，超时会报错。
- W&B 记录源码、配置、最终 PPL、步数、UTC 起止时间、累计耗时及是否超出六小时。

本地不再额外生成 `metrics.jsonl`、`config.json`、`result.json`、`timing.json` 或数据指纹文件。
课程记录器仍会保存、上传源码快照；这部分是明确的课程要求。
已有分词器、数据缓存、模型和历史运行记录不删除。

续训时，在同一个配置字典中填写 `resume`、`resume_from_run` 和
`previous_elapsed_seconds`（从此前关联 W&B 运行读取实际累计耗时，不能清零）。
继续使用原分词器、数据、模型和优化器配置；程序不再自动核对其哈希和配置一致性。
每次启动仍重新编码原文，耗时计入累计预算，不使用免计时的开发缓存。

测试：`python -m pytest`。提交一份 Markdown 文档，包含姓名、学号、两项正式 W&B 链接，
有续训时附关联运行。
