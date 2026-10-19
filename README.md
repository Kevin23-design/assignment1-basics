# BPE 与 Transformer 作业

正式源码包括 `bpe.py`、`train_bpe.py`、`transformer_module.py`、
`transformer_main.py`、`parallel_bpe_encoding.py`、`bpe_memory_runtime.py` 和课程提供的 `wandb_record.py`。
模型核心、BPE 训练/合并规则和课程源码记录器保持不变。

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

任务一入口：

```sh
python train_bpe.py
```

完整文本编码保留 **1,048,576 项、满后整体清空** 的缓存；只缓存不超过 128 字符的
普通预分词片段。正式入口保持课程模板；缓存策略与容量由本人 BPE 源码记录。
流式 `encode_iterable()` 继续使用独立的 2,048 项缓存。

云端三轮缓存探索中，1M 清空策略中位数为 41.499 秒；1M 停止新增为 41.728 秒，
1M LRU 为 51.577 秒，全部通过完整 ID 对照与全文还原。1M 实际缓存 627,462 项，
未触发容量上限，因此本次无法比较它与停止新增策略在满后的行为。当前继续保留原策略。
以上是探索结果，最终提交仍使用正式 W&B 验证记录。

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

模型配置为 4 层、512 维、8 头、FFN 1408、上下文 512、BF16。
batch 64，即每次更新 32,768 token，面向 A100/A800 80GB，尚未在该卡上确认吞吐。
AdamW β=(0.9,0.999)、权重衰减 0.1、梯度裁剪 1.0。
下一版候选基线学习率为 3e-4→3e-5，warmup 500 步，余弦终点为全局 step 65,000；
终点后保持最低学习率。固定预留 600 秒收尾，累计预算仍为 21,600 秒。
模型、batch、AdamW 与 16-worker 编码配置保持不变，从随机权重启动新的正式 run。
这些参数尚待云端验证，65K 不是最终步数保证；本轮评估整套训练安排，不能将收益单独
归因于 warmup、余弦终点或收尾预留。更改学习率或路径直接修改 `CONFIG`。

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

历史成功 run 的完整验证 PPL 为 49.16284，保存与完整验证约 174 秒，为 600 秒预留
提供参考。新 run 必须完整验证并在六小时内结束，再比较最终 PPL、实际最终 step 与吞吐，
在结果出来前不认定已经提优。旧检查点续训应恢复原 run 的 warmup、余弦终点等配置，
不能直接沿用这版新的默认调度。

## 课程要求保留项

- 使用本人 BPE 编码完整原文，不额外插入标记；模型从随机权重开始。
- 编码后写入磁盘 token 数组，从整个训练数组随机抽取连续窗口。
- 模型各层、注意力、交叉熵、AdamW、调度和梯度裁剪均为手写实现。
- 每 100 次更新向 W&B 记录区间训练 loss、PPL、最后使用的 LR 和平均步耗时，尾部也记录。
- 仅保存最后模型及优化器、步数和随机状态到本地 `last.pt`，随后验证完整 OWT 验证集。
- 验证关闭梯度，按总目标 NLL / 总目标数计算 PPL，尾部计分，仅首 token 无预测目标。
- 从入口最顶部计时，导入、W&B、源码、编码、初始化、训练、保存和最终验证均计入累计六小时。
- 候选基线固定预留 600 秒收尾，不自动估算；须在目标设备上验证足够，超时会报错。
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

## BPE 缓存策略探索（云端）

`benchmark_bpe_cache.py` 独立比较 `clear`（满后清空）、`lru`（淘汰最久未使用项）
与 `freeze`（满后停止新增，已有项继续命中），不训练词表、不修改正式 BPE 配置。
默认容量为 64K、256K、1M，每个组合三次正常测速，另可执行独立诊断。

课程服务器使用公共环境，**不要执行 `uv sync` 或安装依赖**：

```sh
source /env/activate_a1.sh
cd /ai/lmfs2026/$USER/assignment1-basics
# 使用自己的 W&B 配置；环境已有登录记录时无需重复登录。
export WANDB_ENTITY="你的实际账号或团队"
export WANDB_PROJECT="lmfs-assignment1"
export STUDENT_ID="你的学号"
export WANDB_MODE="online"
export PYTHONIOENCODING="utf-8"

python benchmark_bpe_cache.py \
  --tokenizer-prefix outputs/830f59c92e15/tokenizer \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --capacities 65536 262144 1048576 \
  --policies clear lru freeze \
  --repeats 3 \
  --diagnostics \
  --output-dir outputs/bpe-cache-exploration
```

分词器前缀须替换为自己的实际路径，三个 JSON 文件必须齐全。输出目录必须不存在，
重复实验请换目录名。默认选择 Linux 当前允许使用的最小 CPU 编号；`--cpu N` 可指定
允许使用的编号。父进程和所有子进程固定在同一个 CPU，串行运行，相关线程环境变量
设为 1，实际最多使用 1 核，低于 16 核上限。公共数据始终以只读方式打开。

完整流程为一次原版参考、27 次正常测速及 9 次独立诊断，共 37 个新子进程。
每次刚加载分词器，从空缓存开始；读取数据、加载、W&B 初始化、源码记录在评分计时前。
计时只包含全文编码、解码与原文比较，ID 精确对照、前 200 token 展示及上传在计时后。
任一还原或 ID 对照失败即停止，查看对应 `.log` 和结果 JSON；正常报告只汇总通过的运行。

默认每个子进程通过课程 `wandb_record.py` 创建探索 run，打印并保存源码快照，记录
特殊 token 配置、前 200 token、全文还原结果及完整耗时。诊断 run 明确标记
`diagnostic=true`，其耗时不纳入正常排名。网络不通时可设 `WANDB_MODE=offline`，
之后使用课程环境的 `wandb sync` 上传相应离线目录。`--no-wandb` 仅供本地开发测试，
其结果不能作为符合课程记录要求的实验材料。

结果目录包含 `manifest.json`（输入/源码 SHA256、环境与固定随机顺序）、逐次 JSON/日志、
`results.json`、`diagnostics.json`、`summary.csv`、`report.md` 和 `integrity.json`。
报告分别按固定容量、固定策略分组，给出中位数与范围。相对原版速度比使用原版单次参考
耗时除以候选中位数，注意单次参考本身的波动。完整 ID 参考以分块二进制文件临时保存，
正常结束或 Python 异常时自动清理；不加载第二份完整 ID 列表。峰值 RSS 为整个子进程
峰值（含导入、W&B），不是缓存单独占用。

源码运行中不要修改。此工具沿用现有 BPE；词表是否来自完整 OWT 训练集、既有核心代码
来源是否符合课程规定，仍需独立确认。本探索报告不替代最终正式 BPE 提交。

仅运行小规模开发测试：

```sh
python -m pytest tests/test_bpe_cache_experiment.py -q
```

## 流式缓存与多进程 BPE 探索（exploration）

`benchmark_parallel_bpe.py` 与 `parallel_bpe_encoding.py` 是独立探索工具，不修改默认
BPE 或正式 Transformer 入口。使用已有 tokenizer 和成功 run 的参考 token 文件；
默认参考为 valid 66,401,098 项 / 265,604,392 字节、train 2,727,120,452 项 /
10,908,481,808 字节。路径必须指向同一分词器生成的完整 OWT 产物，否则比较失败即停止。

服务器激活公共环境并使用自己的 W&B 环境变量，不执行 `uv sync` 或安装依赖：

```sh
source /env/activate_a1.sh
cd /ai/lmfs2026/$USER/assignment1-basics
export WANDB_ENTITY="你的实际账号或团队"
export WANDB_PROJECT="lmfs-assignment1"
export STUDENT_ID="你的学号"
export WANDB_MODE="online"
export PYTHONIOENCODING="utf-8"

CUDA_VISIBLE_DEVICES="" python benchmark_parallel_bpe.py \
  --phase validation \
  --tokenizer-prefix outputs/e9275a3b4a7a/tokenizer \
  --train-file /data/assignment1-data/owt_train.txt \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --reference-train-tokens outputs/6bb24d46a473/train.tokens \
  --reference-valid-tokens outputs/6bb24d46a473/valid.tokens \
  --cache-capacities 2048 65536 262144 1048576 \
  --workers 1 2 4 8 16 \
  --repeats 3 \
  --diagnostics \
  --output-dir outputs/parallel-bpe-exploration
```

输出目录必须不存在。程序读取 Linux 当前允许 CPU，取最多前 16 个；若可用 CPU 少于
请求的最大 worker 数，在启动实验前报错。每轮协调进程使用前 N 个 CPU，worker 各固定
一个 CPU；顶层父进程的线程和已有日志服务子进程也限制到本轮集合。线程环境变量为 1，
GPU 隐藏。采用 spawn，worker 不导入 Transformer、PyTorch 或 W&B。公共原文与参考
文件始终只读，候选与分片只写入个人项目的 `outputs`。

阶段一：单 worker、满后清空，四个缓存容量各三次，诊断另各一次。选择距最快中位数
3% 以内的最小容量。阶段二：选定缓存，1/2/4/8/16 workers 各三次，选择距最快中位数
5% 以内的最少 worker；16 相对 8 核提速不足 5% 时选择 8。两阶段正常测速均使用固定
随机种子的交错顺序，默认共 27 次正常测速和 4 次诊断；诊断不参与选优。

读取方式沿用任务二的 `read(1 << 20)`，表示字符数，不是严格 1 MiB 字节。每次写盘
262,144 个 little-endian uint32 ID，使用标准库 array。扫描从文件起点开始，按分词器
完整特殊 token 清单与优先级匹配，保留跨扫描块的最长特殊 token 缓冲；分隔点只能放在
实际匹配到的 `--split-special-token`（默认 `<|endoftext|>`）后。重复切点合并，产生的
非空范围不足 N 个时该配置标记 unavailable，不参与选优，不静默减少 worker 数。

计时从启动新的协调子进程前开始，到它完成拼接退出为止；涵盖协调进程启动、分词器加载、
切点扫描、worker 启动/加载/读取/编码写盘/退出和拼接。`encode_seconds` 是所有 worker
编码阶段的墙钟时间，包含 worker 启动、加载和读写。分项之和与总时间可能有差异，因为
总时间还包括协调进程启动及少量管理操作。输入指纹、完整 ID 比较、字节还原、候选哈希、
报告与 W&B 操作均在计时外。单 worker 不做额外二进制拷贝。

每个候选先核对已知大小，再与参考文件分块逐字节比较；另外将 token 的原始词表 bytes
分块拼接，与输入原始 bytes 对照。任何不一致立即停止。验证候选核验后删除。worker
失败时终止并 join 其他 worker，清理分片、`.partial`，不生成最终候选；查看相应 `.log`。

顶层创建一个 `transformer-bpe-parallel-exploration` W&B run，课程记录器打印、保存
完整源码及配置，Table 汇总全部候选与诊断；子进程不创建 run。输出含 `manifest.json`、
逐次 job/结果/日志、`schedule.json`、`results.json`、`results.csv`、`report.md`、
`selection.json`、`integrity.json` 及 W&B 链接。每轮进程重新加载分词器，从空缓存开始。
`parent_peak_rss` 指协调进程峰值，worker 峰值之和不等于同时发生的实际内存峰值。

阶段三单独启动，继承并校验验证阶段的路径、输入和源码 SHA-256，取验证集最快的两种
有效 worker 配置各编码完整训练集一次。因此两阶段之间不能修改或切换源码版本。

```sh
CUDA_VISIBLE_DEVICES="" python benchmark_parallel_bpe.py \
  --phase train-confirmation \
  --validation-report outputs/parallel-bpe-exploration/selection.json \
  --output-dir outputs/parallel-bpe-train-confirmation
```

训练确认要求输出所在磁盘至少约 33 GB 空闲；正常结束后只保留最终候选 `train.tokens`，
非最终候选删除，哈希和报告保留。最多同时保留一份已有候选与一份正在生成的候选。
选择距训练集最快耗时 5% 以内的最少 worker 数，适用时继续执行 8/16 核提速规则。
可在 `report.md` 与 `selection.json` 查看结果；不可用配置单独列出，不作为速度优势。

网络中断可用 `WANDB_MODE=offline`，随后同步离线 run。`--development --no-wandb`
仅供小文本测试，允许从小参考文件推导 token 数及在 macOS 跳过 affinity；它的结果不能
作为云端课程证据。开发参数在报告中明确记录。

```sh
python -m pytest tests/test_parallel_bpe_experiment.py -q
```

本工具不自动接入正式 Transformer，也不预生成 token 用于绕过正式计时。正式接入需等待
结果确认，仍应从入口开始计时、重新编码 train/valid，并在训练前确认所有 worker 已退出。
完整训练集来源和课程对 AI 核心代码来源的规定须独立确认。

## 四 GPU 上下文 × 峰值学习率探索

本节仅用于 `exploration`。训练与编码同步最新 `dev` 基线，已有 BPE 探索工具保留，
不修改 `dev`。四组固定配置如下，每步均处理 32,768 个目标 token：

| 组 | CUDA_VISIBLE_DEVICES | 训练/验证上下文 | batch | 峰值 LR |
|---|---:|---:|---:|---:|
| A | 0 | 512 | 64 | 3e-4 |
| B | 1 | 512 | 64 | 6e-4 |
| C | 2 | 256 | 128 | 3e-4 |
| D | 3 | 256 | 128 | 6e-4 |

其余固定为 4 层、512 维、8 头、FFN 1408、BF16、seed 42、warmup 500、余弦终点
65K、最低 LR 3e-5、AdamW β=(0.9,0.999)、weight decay 0.1、梯度裁剪 1.0。
每组独立六小时预算，固定 600 秒收尾；均从随机权重开始，无续训或超参覆盖入口。
本轮**不做显存试训或预检**，OOM 会作为该组失败记录。

本轮以用户已获批的 **64 个逻辑 CPU 总配额**运行，不沿用单个 run 的 16 核总配额
来解释四组并发权限。启动器读取 Linux 当前允许 CPU，取前 64 个，分成四个互不重叠的
16 核集合；不足 64 核则启动前报错。每组父进程与子进程均硬限制在对应集合，编码采用
16 workers、每 worker 1M 缓存、满后清空，worker 内部线程为 1 且隐藏 GPU。
正式进程的计算线程最多 16 个，只看到指定的一张 GPU，内部设备为 `cuda:0`。
启动器仅使用标准库，不导入 NumPy/PyTorch，不创建第五个 W&B run。

云端进入自己的 exploration 项目目录，激活共享环境并配置个人 W&B：

```sh
source /env/activate_a1.sh
export WANDB_ENTITY="你的实际账号或团队"
export WANDB_PROJECT="lmfs-assignment1"
export STUDENT_ID="你的学号"
export WANDB_MODE="online"
export PYTHONIOENCODING="utf-8"

python launch_context_lr_experiments.py \
  --gpus 0 1 2 3 \
  --output-dir outputs/context-lr-2x2
```

请确保输出目录不存在，磁盘至少有约 100 GB 空闲，公共 OWT 文件可读，项目内
`outputs/e9275a3b4a7a/tokenizer` 的三个 JSON 文件齐全。公共环境只读使用，不安装依赖。
启动器分别生成 `A.log` 至 `D.log`，以及四个独立目录 `A/` 至 `D/`。W&B 仍在
`outputs/<run_id>/source` 保存源码快照。数据源、tokenizer、模型、优化器和随机种子保持
一致；四组均重新编码完整原文到各自目录，不共享可写 token 文件。编码、加载、读写、
初始化、W&B 源码记录、训练、保存和完整验证全部计入各自入口起算的六小时。
各组全部编码 worker 退出后才初始化模型。

实验系列 ID 默认由启动时间生成，也可用 `--study-id my-study` 指定。每组创建一个
`transformer-context-lr-exploration` W&B run，名称包含系列与 A/B/C/D，group 为
学号加系列 ID。源文件必须在项目目录内，运行中不要修改或切换源码。

如需单独启动指定组，使用同一个系列 ID，显式提供其获批的 16 个 CPU 编号。例如下例
仅适用于 CPU 0..15 位于当前允许集合时：

```sh
CUDA_VISIBLE_DEVICES=0 python train_context_lr_experiment.py \
  --case A --cpu-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --study-id my-study --output-dir outputs/my-study-A
```

沿用每 100 步及尾部区间的 loss/PPL/LR/平均步耗时日志。新增 `train/target_tokens`
作为累计目标 token 坐标，W&B 可将其选为 loss 曲线横轴；相同 step 对应相同目标 token
数，但上下文改变后并不使用相同的采样序列。Summary 记录整体训练 token/s、最终 step、
完整验证 loss/PPL、阶段耗时、实际 GPU 型号与容量、GPU 峰值 allocated/reserved bytes。
GPU 峰值从初始化模型前重置，包含模型、训练和验证分配；无额外逐步 `.item()`。

A/B 使用完整 512 窗口验证，C/D 使用完整 256 窗口验证，eval batch 均为 16。
所有结果要求有效目标数 **66,401,097**，尾部计分，只排除首 token。**不做统一窗口验证**。
B/A 和 D/C 比较同上下文的学习率；C/A、D/B 比较固定 token batch 的上下文安排。
跨上下文的最终 PPL 评价整套配置，不能单独归因于训练上下文。四组并发共享磁盘，
编码耗时可能不同于单 GPU 历史记录，应结合实际训练 token 预算和 GPU 型号解释吞吐。

每组可捕获的 OOM、非有限 loss、超时或验证失败单独记录，不终止其他组；有限性检查
保持现有区间频率，不是每一步进行数值读取。单组 `result.json` 保存 run 链接、配置、
状态与 Summary；启动器在组结束后更新 `results.json`、`results.csv` 与 `report.md`。
只有完整验证成功、源码未变、验证目标数正确且累计不超过六小时的结果进入有效表。
历史完整验证 PPL **40.60899** 仅作参考，本工具不会自动修改 dev。

按 Ctrl-C 或对启动器发送 SIGTERM，会终止并等待它管理的 run 进程组及嵌套编码组。
单独对某组发送 SIGTERM 只停止该组；强制 SIGKILL/断电时可能没有完整的末尾指标，
对应结果不作为有效成绩。网络不通可设 `WANDB_MODE=offline`，之后分别同步四个离线 run。

本地仅运行小规模测试，不启动云端 GPU 训练：

```sh
python -m pytest tests/test_context_lr_experiments.py tests/test_training.py \
  tests/test_optimizer.py tests/test_train_bpe.py tests/test_tokenizer.py -q
```

## GPU 2、3：优化器组合与大 batch 候选

新增 `optimizer-batch` suite，仅用于 `exploration`；不更新 `dev`。旧 A–D 配置和默认
四卡启动命令保持兼容。新 suite 固定为：

| 参数 | E（GPU 2） | F（GPU 3） |
|---|---:|---:|
| 训练/验证 context | 256 | 256 |
| batch | 128 | 256 |
| 每步目标 token | 32,768 | 65,536 |
| 峰值 LR | 3e-4 | 6e-4 |
| AdamW β | (0.9, 0.95) | (0.9, 0.95) |
| weight decay | 0.01 | 0.01 |
| 梯度裁剪阈值 | 2.0 | 2.0 |
| warmup steps | 500 | 250 |
| cosine 终点 step | 65,000 | 32,500 |

两组最低 LR 均为 3e-5。F 的 warmup 和余弦终点步数减半，与 E 对齐到相同目标 token
预算：warmup **16,384,000**、余弦终点 **2,129,920,000** tokens。模型、BF16、seed 42、
初始化、BPE、原文、采样规则和 eval batch 16 沿用当前基线；均从随机权重开始，各自六小时，
固定 600 秒收尾，重新编码完整 train/valid。不做显存试训、不自动降低 batch；OOM 单独记为
失败，不停止另一组。

F 使用 6e-4 的依据是上一轮 256 窗口实验：C（batch128、LR3e-4）的完整验证 PPL 为
45.18785，D（batch128、LR6e-4）为 41.75639。F 是性能候选，不是 batch 单因素实验。
E 同时改变三项优化器设置，F 又改变 batch 与峰值 LR，本轮只评估组合效果，不提前认定更优。

运行（共享环境和个人 W&B 变量沿用上一节）：

```sh
source /env/activate_a1.sh

python launch_context_lr_experiments.py \
  --suite optimizer-batch \
  --gpus 2 3 \
  --output-dir outputs/optimizer-batch-2gpu
```

新 suite 不指定 `--gpus` 时也默认 2、3，不指定输出路径时默认上述目录。输出目录必须不存在，
磁盘至少约 **50 GB** 空闲。旧 `context-lr` suite 仍默认 GPU 0/1/2/3、100 GB 磁盘检查。
单组入口可用 `--case E` 或 `--case F`，suite 自动推断，也可显式传入
`--suite optimizer-batch`；不能传入任意超参或不匹配的 case/suite。

CPU 保留原 C/D 分区：从当前允许 CPU 排序后的前 64 核中，E 使用第 33–48 核，F 使用
第 49–64 核。因此仍要求允许集合至少有 **64 核**，但两组实际只使用后面的 **32 核**，
不自动改用前 32 核。每组维持 16 个编码 worker、每 worker 1M 缓存、满后清空、单内部线程、
worker 隐藏 GPU；其余进程资源隔离和异常清理沿用原启动器。
若 GPU 0/1 仍在同目录运行旧版本，应把新版本部署到独立项目副本，避免源码变化使旧 run
的源码完整性检查失败；保留 CPU 分区不等于可以在运行中修改其源码。

W&B job type 为 `transformer-optimizer-batch-exploration`，每组一个 run，共用系列 ID。
`result.json` 和 W&B config 包含 suite、完整超参、每步 token、warmup/余弦终点 token。
汇总 CSV/Markdown 新增 β₁/β₂、weight decay、裁剪阈值、每步 token 和调度终点；报告按
各组实际 token 数显示，不再将所有实验写成每步固定 32,768 tokens。

上一轮提前约 1800 秒退出的原因尚未确认，本轮不修改停训策略。触发原条件
`now + reserve_seconds + 2 * last_step_seconds >= deadline` 时，在 Summary 记录：

- `time/train_stop_reason`：`reserve_plus_two_last_steps`；
- `time/train_stop_elapsed_seconds`：入口累计 elapsed（含续训历史预算）；
- `time/train_stop_remaining_seconds`：距离本轮累计截止时刻的剩余秒数；
- `time/train_stop_last_step_seconds`：最后一次已完成更新的耗时；
- `time/train_stop_step_safety_seconds`：上述耗时的两倍；
- `time/train_stop_reserve_seconds`：本轮固定收尾预留。

这些字段连同编码、训练、保存和验证耗时写入汇总 CSV。异常退出使用已有的失败状态记录，
不把它伪装成正常触发收尾条件。诊断只记录现有 CPU 时钟，不新增逐步 GPU `.item()`。

两组均用完整 256 窗口验证，目标数必须为 **66,401,097**。主要比较最终验证 PPL，另外按
相同累计目标 token 附近比较区间 loss，结合实际 token/s、最终 token 总量、峰值显存和预算
利用情况判断。E/F 的每步 token 数不同，不按相同 step 假定预算相同，也不将不同并发负载下
的吞吐差异全部归因于超参。仅完整验证成功、源码未变且累计不超过六小时的结果进入有效表。

小规模本地检查（不启动正式 GPU 训练）：

```sh
python -m pytest tests/test_optimizer_batch_suite.py tests/test_context_lr_experiments.py \
  tests/test_training.py tests/test_optimizer.py tests/test_train_bpe.py tests/test_tokenizer.py -q
```

## 任务一：完整 encode/decode/比较效率探索

`benchmark_task1_efficiency.py` 与 `task1_memory_encoding.py` 是独立实验工具。
不修改默认 `BPE.encode()`、`decode()`、流式接口或任务二写盘流程；复用已有 tokenizer，
不重新训练词表。此实验输出最终内存 `list[int]`，总耗时包含全文 encode、全文 decode
和原文精确比较。任务二的编码写盘成绩不能作为本实验成绩。

云端激活公共环境、配置自己的 W&B 后，在 exploration 项目目录执行：

```sh
source /env/activate_a1.sh

python benchmark_task1_efficiency.py \
  --tokenizer-prefix outputs/e9275a3b4a7a/tokenizer \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --workers 1 2 4 8 16 \
  --repeats 3 \
  --diagnostics \
  --output-dir outputs/task1-efficiency
```

输出目录必须不存在。默认使用当前允许的前 16 个逻辑 CPU；不足请求的 worker 数时
直接报错，不自动降配。可以加 `--cpu-ids` 指定 16 个允许且空闲的 CPU 编号，避免与正在
训练的实验共用 CPU；实验期间也应避免其他高负载任务影响比较。GPU 隐藏，所有底层线程
变量为 1，父进程、后台记录服务及 worker 限制在所选 CPU 集合内，整个实验最多 16 核。
标准单进程基线使用其中首个 CPU，并行配置使用前 N 个。资源数量会记录到结果中。

计时前：加载新的父分词器、读取完整 UTF-8 文本（保留 CR/LF）。计时内：计算安全切点、
启动全新 spawn workers、加载 worker 分词器、文本复制/IPC、普通 BPE 编码、uint32 ID
返回、有序组装完整 Python list、worker 退出、全文解码、原文比较。缓存均为 1M、满后
清空、仅缓存不超过 128 字符的片段，从空开始；不在计时前建池、分片或编码验证集。
同一 worker 在本次验证内处理后续任务时可以自然使用已有缓存。

分片直接在已加载文本中匹配完整特殊 token 清单，遵守生产实现的匹配优先级，只在真实
分隔 token 后切分。默认 `--separator '<|endoftext|>'`，它须在 tokenizer 配置中。
包含关系不会通过任意子串搜索处理。重复切点合并，不足所需非空分片时标记 unavailable，
不按任意字符或字节截断普通片段。

自动分阶段执行：

1. 原版单进程 S 与 P1/P2/P4/P8/P16，各三次，固定种子交错，共 18 次完整验证。
   按总耗时中位数选优，距最快 3% 以内优先较少 worker，单进程与 P1 接近时优先原版。
2. 并行胜出时，固定 worker 数，比较每 worker 1/2/4 个任务，各三次；单进程胜出则跳过。
   本阶段以最快中位数选择分片粒度。
3. 固定编码方案，比较原版解码与 64K/256K/1M 个 ID 的 bytes 分块拼接，各三次。
   所有 bytes 块拼接后仅做一次 UTF-8 decode，不能逐块独立解码。以完整总耗时中位数选优。
4. `--diagnostics` 另跑原版和选定配置，记录缓存查询/命中/清空，并生成 parent/worker
   `.prof` 文件。诊断时间不排名。最后在新进程、空缓存下做一次独立确认。

默认流程在并行胜出时共 40 次正常比较/确认，加一次独立参考生成，另有最多两次诊断；
单进程胜出时跳过 9 次分片粒度比较。所有候选串行运行，不并发相互抢资源。

正常计时外，对每个候选与参考进行完整 uint32 ID 对照，要求 **66,401,098** 个 ID 全部
一致，并生成前 200 token 的 ID、hex、逐项解码、字节位置和匹配信息。任一还原失败或
ID 不一致立即停止。默认参考由独立子进程的原版 BPE 生成，临时参考文件在结束或 Python
异常时清理；参考生成行的 ID 对照字段为空，表示尚不适用。参考实例缓存不参与候选运行。
已有可信且与当前数据/分词器匹配的参考时，可附加：

```sh
--reference-tokens outputs/6bb24d46a473/valid.tokens
```

参考文件只读，启动时验证大小，记录 SHA-256；候选仍全部逐项精确比较。运行过程中只在
计时外读写参考，不使用 token 写盘途径返回候选。进程失败或用户中断时会终止并等待编码
worker；外层子进程组提供额外清理。公共原文和 tokenizer 始终只读。

一个 `bpe-task1-efficiency-exploration` W&B run 保存完整源码和 Table。Summary 中
`bpe/valid_seconds` 是**独立确认**的完整耗时，不是从重复实验中挑出的最短单次值。
前 200 token 也来自该确认。每个候选的分阶段、总耗时、正确性、CPU、RSS 均记录到本地
结果；分片/启动/编码及 IPC/组装分项位于逐次 JSON 和 CSV。编码与传输重叠的阶段不能
解释为纯 BPE 计算时间；worker RSS 之和不是同时发生的实际峰值。

输出目录包含输入/源码指纹与环境的 `manifest.json`、参考信息、执行顺序、逐次 job/结果/
日志、`results.json`、`results.csv`、`report.md`、`selection.json`、`integrity.json` 和
W&B 链接。诊断 `.prof` 可用标准库 pstats 或 cProfile 工具单独分析；不把诊断时间当成绩。
若源码或输入前后 SHA-256 变化，整个探索标记失败。

本轮只比较既定词表的实现效率，不证明词表来自完整 OWT 训练集；
`submission_provenance_verified=false` 明确保留此限制。完成探索后再决定是否接入正式
任务一入口，不自动替换实现，不将本工具当成完整训练与提交来源证明。

本地小文本测试（不运行完整 OWT）：

```sh
python -m pytest tests/test_task1_efficiency.py tests/test_train_bpe.py \
  tests/test_tokenizer.py tests/test_parallel_bpe_experiment.py -q
```

`--development --no-wandb` 仅供小 fixture，可在 macOS 跳过 affinity；该模式不作为课程
实验材料，不修改公共环境依赖。

### 下一轮：任务一局部精调与配对确认

在 `exploration` 使用同一脚本的 `--suite local-refinement`；不改 `dev`、生产 BPE 或
任务二入口。当前对照固定为 16 workers、每 worker 平均 4 个任务（共 64 分片）、
262,144 IDs 解码块、每 worker 1M 清空缓存。上轮完整流程中位数 12.544 秒、独立复测
12.561 秒只作历史参考，选优使用本轮重新测量的对照。

```sh
source /env/activate_a1.sh

python benchmark_task1_efficiency.py \
  --suite local-refinement \
  --tokenizer-prefix outputs/e9275a3b4a7a/tokenizer \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --diagnostics \
  --output-dir outputs/task1-refinement
```

输出目录须不存在。沿用前述最多 16 个逻辑 CPU 的硬限制、GPU 隐藏、线程数 1、独立新
进程/空缓存、公共数据只读、完整 ID 精确对照和全文还原检查。需要避开其他任务时用
`--cpu-ids` 指定 16 个允许且空闲的 CPU。云端不允许降到较少 worker 或少于五轮；
自定义搜索范围时必须包含当前解码块 262,144 与每 worker 4 个任务作为对照。

默认依次执行：

1. 固定 16 workers 和 64 分片，比较解码块 65,536 / 131,072 / 262,144 / 524,288 IDs，
   各 5 次，逐轮固定种子打乱顺序，共 20 次。按完整流程中位数选择解码块。
2. 固定选出的解码块，比较每 worker 平均 2 / 4 / 8 / 16 个任务，即 32 / 64 / 128 / 256
   个安全分片，各 5 次，共 20 次。按完整流程中位数产生候选，分片不足的配置不参与选优。
3. `--diagnostics` 对当前对照和候选各跑一次，生成 parent/worker `.prof` 及逐任务诊断。
   即使候选与对照相同，也保留这两次独立诊断，诊断均不参与速度排名。
4. 在新进程和空缓存下，当前对照与候选再配对运行 5 轮；每轮随机决定先后顺序，共 10 次。
   只有候选总耗时中位数比对照降低至少 3%，且至少赢得 4/5 对，才推荐候选；否则保留当前
   配置。配置相同时始终保留当前方案，不把测量噪声解释为改进。

共 50 次正常搜索/确认、2 次可选诊断；未提供 `--reference-tokens` 时，另有一次原版
参考生成。正式数据的 ID 数必须为 66,401,098，任何 ID 或还原不一致立即停止。
正常计时始终是完整 encode → 完整 `list[int]` → decode → 原文比较；所有分片、启动、
worker 加载/退出、通信和列表组装均在其中，不提前建池或预编码。

独立诊断新增 worker 每个任务的字符数、token 数、编码、uint32 打包、send 和取任务等待
耗时，以及 worker 完成时间差、父进程等待/接收和 join 耗时。send 含序列化和管道阻塞，
receive 含等待和反序列化，不能视为纯通信成本，各进程的重叠耗时也不能相加。
解码诊断分别测字节查找、块内拼接、最终拼接和 UTF-8 转换；为分离这些阶段会额外物化
每块字节引用，因此其分项仅辅助定位，不代表正常解码的准确耗时分解。正常测速不走此路径。

一个新的 `bpe-task1-efficiency-refinement` W&B run 保存源码、完整配置、候选 Table、诊断
JSON 和 `.prof`。本地报告包含正常测速的中位数/范围/内存、诊断表和配对决定。
`selection.json` 的 `selections.paired_confirmation` 保存每对耗时、改善幅度、获胜数和
选优原因；只有正确性通过且源码/输入指纹未变才会写出完成结果。
`bpe/valid_seconds` 与前 200 token 取最终推荐配置在第 5 对中的实际运行，绝不挑最短值；
配对中位数另列。实验不会自动把推荐配置接入正式入口。

本地仅运行小文本和编排检查：

```sh
python -m pytest tests/test_task1_refinement.py tests/test_task1_efficiency.py \
  tests/test_train_bpe.py tests/test_tokenizer.py tests/test_parallel_bpe_experiment.py -q
```

## 任务一：正式接口接入层探索（B / R / S / D）

`exploration` 选择性同步正式 run `8b68b68e0cbe` 对应的 BPE、运行辅助模块及课程原样
`train_bpe.py`，保留已有探索工具。新实验只复用词表，不训练，不改 `dev`，不改合并规则。
旧缓存/单核工具显式调用串行核心和原有全文 bytes 解码；旧并行 helper 的 worker 也显式
串行，防止新版 BPE 的自动并行改变历史对照或启动嵌套 worker。

```sh
source /env/activate_a1.sh

python benchmark_task1_transport.py \
  --tokenizer-prefix outputs/8b68b68e0cbe/tokenizer \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-transport-exploration
```

输出目录必须不存在且位于个人项目 `outputs`。可选
`--reference-tokens outputs/6bb24d46a473/valid.tokens`；未提供时，独立串行进程生成临时
uint32 参考，在正常退出或 Python 异常时删除。参考文件只读，记录 SHA-256 和长度。
分词器前缀的三个 JSON 必须存在；默认前缀是本次正式运行保存的词表。

固定参数：**16 workers、256 个安全分片、每 worker 1M 满后清空缓存、131,072 IDs 解码块**。
所有正常测速调用 `BPE.encode(text)` → 完整 `list[int]` → `BPE.decode(ids)` → 原文比较。
输入读取和父分词器加载在计时前；分片、输入/共享内存准备、状态传输、进程启动/退出、
完整列表组装和资源释放均计入；ID 对照、前 200 token 及上传在计时后。
各次使用全新试验子进程和空缓存，不保留进程池。试验进程计时前导入 W&B 以匹配课程
调用环境，但不创建自己的 run；worker 的独立轻量入口不导入 W&B、NumPy 或 PyTorch。
为选择实验后端，运行模块在计时前统一导入，manifest 明确记录；不提前准备数据或启动
worker。最终确认仍属于本探索的调用环境，不替代原样课程入口的正式运行成绩。

| 后端 | 正常路径 |
|---|---|
| B | 未改动的正式 runtime：全文传给协调进程，全部 worker 退出后转发结果。 |
| R | 相同输入路径；协调进程每收到一个分片就转发，调用者接受乱序并按原顺序组装。 |
| S | 共享输入与每 worker 一个可复用 uint32 输出槽，协调进程只转发控制描述信息。 |
| D | 调用进程直接管理独立轻量 Python worker；每个 worker 至多一个在途任务，动态分发、直接收取结果。 |

S 按既有安全字符分片分别进行 UTF-8 编码，形成共享输入和字节范围，worker 严格解码自己
的范围。每个输出槽容量为最大分片原始 UTF-8 字节数的四倍：token 数不超过原始字节数，
**不依赖参考 token 数预分配**。调用者复制结果后 ACK，收到 ACK 后才可复用对应槽；乱序
分片保存在调用者自身内存中。所有共享段由调用者创建、关闭和 unlink，其他进程仅以
`track=False` 附加。S 要求 Python 3.13+；在候选自身计时范围内检查 `/dev/shm` 可用空间，
实际输入和输出槽需求之外预留 64 MiB。版本或容量不满足时记录 `unavailable`，不偷偷
回退成管道版本；已启动后的 worker 异常仍中止整个实验。

全程在同一 CPU 集合串行比较，默认取允许的前 16 个逻辑 CPU；不足 16 核直接报错。
可用 `--cpu-ids` 指定恰好 16 个允许且空闲的编号。父进程、记录服务、协调进程和 worker
都限制在这组 CPU 内，每个 worker 固定一个，底层线程为 1、GPU 隐藏；不使用 `fork`。
原文与参考始终只读，正常候选不产生 token 文件，共享内存无持久 token 产物。

默认流程（固定种子逐轮交错）：

1. 先对 B 做一次独立诊断，不参与排名。
2. B/R/S/D 各测 5 次，共 20 次；S 不可用时记录原因并排除其选优资格。
3. 对中位数最快的候选另做一次独立诊断。
4. 候选与 B 再做 5 对全新进程复测，每对随机先后顺序，共 10 次。
5. 候选必须中位数改善至少 3%，且赢得至少 4/5 对，才推荐替换；否则保留 B。
   候选仍为 B 时不把噪声当成改进。
6. 对最终推荐方案另做一次独立确认；W&B `bpe/valid_seconds` 取这一次实际完整耗时，
   不取历史值、中位数或最短单次作为确认成绩。

完整默认流程为 **31 次正常测速 + 2 次诊断**；未提供参考文件时再加一次串行参考生成。
每个成功候选必须恰好返回 **66,401,098** 个 ID，通过全部 ID 精确对照与全文还原；
任一正确性失败立即停止。进程异常或 SIGINT/SIGTERM 时回收所属进程组，并释放共享段，
不返回部分列表。源码/输入指纹前后不一致时不生成有效选优结论。

诊断默认不启用 cProfile，单独记录安全分片、共享准备、启动、请求发送、首结果到达、
等待/接收、列表组装、worker 编码/打包/发送、完成时间差、退出、共享释放以及解码/比较。
保存各 worker 实际任务分布、affinity、线程/GPU 环境和内存；send/receive 含序列化、
阻塞或等待，不是纯通信成本，跨进程重叠时间不能相加。逻辑载荷统计按 UTF-8/uint32
计算，不冒充实测 pickle 线上字节数。S 和 D 是整套实现比较，调度也有差异，不能将全部
收益归因于减少复制。B 正常路径不增加探针，worker RSS 不可得时留空；独立 B 诊断使用
同样的屏障/转发结构加阶段探针，诊断耗时不当成正常测速。

一个 `bpe-task1-transport-exploration` W&B run 保存完整源码、配置、候选 Table 和逐次
结果。个人输出目录保存 `manifest.json`、指纹、顺序、逐次 job/日志/结果、参考信息、
`results.json`、`results.csv`、`report.md`、`selection.json`、`integrity.json` 和 W&B 链接。
推荐结论与配对明细见 `selection.json`，失败原因见 `failure.json` 和对应 trial 日志。
18.437 秒与 12.012 秒只作历史参考，本轮与重新测量的 B 配对，不自动更新 `dev`，也不
承诺恢复到 12 秒；本探索不证明原词表的训练来源。

本地仅小文本与回归测试，不执行完整 OWT：

```sh
python -m pytest tests/test_task1_transport.py tests/test_task1_efficiency.py \
  tests/test_task1_refinement.py tests/test_bpe_cache_experiment.py \
  tests/test_parallel_bpe_experiment.py tests/test_tokenizer.py tests/test_train_bpe.py -q
```

`--development --no-wandb --repeats 1` 仅供小 fixture，默认两个 worker（可用
`--development-workers` 调整），允许 macOS 跳过 Linux affinity；不作为课程性能材料。

## 下一轮任务二：最佳配置、学习率与参考仓库适配

`exploration` 新增 `best-lr-reference` suite，保留旧 A–F 入口与配置，`dev` 不变。
当前最佳有效参考为 B 的完整验证 PPL **40.18098**（512 窗口）。E/F 的 PPL 为
47.00493 / 43.51419，batch 翻倍时吞吐仅提高约 2.7%，本轮不继续扩大 batch。

| 参数 | G：最佳 B 复测 | H：提高峰值 LR | I：参考仓库六小时适配版 |
|---|---:|---:|---:|
| GPU | 1 | 2 | 3 |
| 层数 / d_model | 4 / 512 | 4 / 512 | 4 / 512 |
| 注意力头数 / FFN | 8 / 1408 | 8 / 1408 | 16 / 1344 |
| 训练与验证 context / batch | 512 / 64 | 512 / 64 | 256 / 128 |
| 每步目标 token | 32,768 | 32,768 | 32,768 |
| 峰值 / 最低 LR | 6e-4 / 3e-5 | 1e-3 / 3e-5 | 1e-2 / 1e-3 |
| warmup / 余弦终点 step | 500 / 65,000 | 500 / 65,000 | 6,500 / 65,000 |
| AdamW β | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.95) |
| weight decay / 裁剪 | 0.1 / 1.0 | 0.1 / 1.0 | 0.01 / 1.0 |
| seed | 42 | 42 | 1337 |

共同设置为 BF16、RoPE θ=10,000、AdamW ε=1e-8、eval batch 16、六小时预算和
600 秒收尾预留。调度终点仍指全局 step 65,000，此后保持最低 LR，不按步数提前结束。
G/H 只改变峰值 LR；I 同时改变多项条件和 seed，只评估整套配置。

参考仓库固定版本 `c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825`：
[OWT 扫描配置](https://github.com/donglinkang2021/cs336-assignment1-basics/blob/c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825/scripts/openwebtext_learning_rate.sh)、
[实际训练入口](https://github.com/donglinkang2021/cs336-assignment1-basics/blob/c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825/train.py)。
I 选择扫描中的 LR0.01、β₂0.95、WD0.01；裁剪按实际代码 **1.0**，不是脚本声明的 2.0。
原本 10K 步调度扩展到 65K，保留 10% warmup 和最低 LR 比例。保留本人 BPE、BF16、
现有初始化与采样、完整验证，不增加中间验证或按最佳中间成绩挑选检查点。
仅参考配置，不移植核心实现；此组不称为完全复现。来源、固定版本和适配差异写入 I 的
W&B 配置以及本地 manifest/results/report。

云端运行（预先配置个人 W&B 环境变量）：

```bash
source /env/activate_a1.sh

python launch_context_lr_experiments.py \
  --suite best-lr-reference \
  --gpus 1 2 3 \
  --output-dir outputs/best-lr-reference-3gpu
```

默认 GPU 与上述命令相同。CPU 分配沿用排序后允许 CPU 的前 64 核：G 使用第 17–32 核，
H 第 33–48 核，I 第 49–64 核，共互斥使用 48 核；不足 64 个允许 CPU 直接失败。
每组硬限制 16 核，各自只看到一张 GPU，内部为 cuda:0。编码保持 16 workers、每 worker
1M 清空缓存、内部线程 1、GPU 隐藏；所有编码 worker 退出后才初始化模型。
输出目录必须不存在，磁盘预检至少 **75 GB**；公共数据只读，不修改共享依赖。

各组从自己的入口顶部计时，重新用 `outputs/e9275a3b4a7a/tokenizer` 编码公共完整 OWT
原文，从随机权重训练，不复用历史 token 文件或检查点。每组独立日志、产物和 W&B run，
job type 为 `transformer-best-lr-reference-exploration`，共用配置中的 experiment_series；
W&B group 保持课程记录器的学号分组，不新增启动器 run。
不做 GPU 试训，不自动降 batch；单组 OOM、非有限 loss、验证失败或超时不停止其他组。
中断启动器会终止并回收所属进程，保留原截止检查及停训诊断。

沿用区间 loss/PPL/LR、累计目标 token、吞吐、峰值显存、各阶段和总耗时记录。
每组要求完整验证 **66,401,097** 个目标，尾部计分。G/H 的 512 窗口结果和 I 的
256 窗口结果分表展示，不混成同口径排名。I 可参考历史同窗口 C/D/E/F，但差异不能
归因于独立参数；相同累计 token 附近的区间训练 loss 仅为辅助证据。
仅源码未变、完整验证成功且累计不超过六小时的结果有效；不声称单次结果统计显著，
不自动修改 dev，也不提前认定 LR0.01 有效。

本地仅运行小规模测试：

```bash
python -m pytest tests/test_best_lr_reference_suite.py tests/test_context_lr_experiments.py \
  tests/test_optimizer_batch_suite.py tests/test_training.py tests/test_optimizer.py \
  tests/test_tokenizer.py tests/test_train_bpe.py tests/test_parallel_bpe_experiment.py -q
```

## GPU 1 单组 J：峰值学习率 2e-3

`exploration` 新增固定 `peak-lr-single` suite，只包含 J，旧 A–I 和 `dev` 不变。
历史 H（run `2cbae6a7f35e`）的 512 窗口完整验证 PPL 为 **36.14949**，相对 G 的
38.10523 降低约 5.13%；两组训练目标 token 数仅差约 0.49%。本轮只将 H 的峰值 LR
从 **1e-3 提高至 2e-3**，评估更高 LR 能否继续改善六小时内的最后模型。

| 参数 | 固定 J 配置 |
|---|---|
| GPU / 内部 device | 1 / cuda:0 |
| 模型 | 4 层、512 维、8 头、FFN 1408、RoPE θ=10,000 |
| 训练与验证 context / batch | 512 / 64，每步 32,768 目标 token |
| 峰值 / 最低 LR | 2e-3 / 3e-5 |
| warmup / 余弦终点 step | 500 / 65,000 |
| AdamW β / ε / weight decay | (0.9, 0.999) / 1e-8 / 0.1 |
| 梯度裁剪 / 精度 / seed | 1.0 / BF16 / 42 |
| eval batch / 完整验证目标数 | 16 / 66,401,097，尾部计分 |
| 总预算 / 收尾预留 | 21,600 / 600 秒 |

65K 后保持最低 LR，继续按六小时预算训练，不固定为历史 H 的 54,940 步。
只保存最后模型，不做中间验证或选择最佳检查点。重新使用现有
`outputs/e9275a3b4a7a/tokenizer` 编码完整公共 train/valid 原文，从随机权重开始；
不续训 H，不复用历史 token 文件或模型检查点。

云端运行（预先配置个人 W&B 环境变量）：

```bash
source /env/activate_a1.sh

python launch_context_lr_experiments.py \
  --suite peak-lr-single \
  --gpus 1 \
  --output-dir outputs/peak-lr-2e3-single
```

默认 GPU 为 1。使用当前允许 CPU 排序后的前 **16** 个，父进程及 worker 均受 affinity
硬限制；不足 16 核直接失败。新 suite 不要求 64 核，旧 suite 的 64 核分区要求不变。
编码继续使用 16 workers、每 worker 1M 清空缓存、内部线程 1、GPU 隐藏；全部退出后
才初始化模型。输出目录必须不存在，磁盘预检至少 **25 GB**。公共数据只读，不修改
共享依赖，不做额外 GPU 试训，不自动降低 batch 或换参数重跑。

只创建一个 `transformer-peak-lr-single-exploration` W&B run，课程记录器保留完整源码，
入口导入、记录、编码、训练、保存和完整验证全部计时。沿用低频有限性检查、截止检查、
停训诊断和异常清理；OOM、非有限 loss、超时或验证失败记为失败。

W&B 配置、manifest 和汇总 JSON 保留历史 H 的 run 链接与指标。本地 JSON/CSV/Markdown
记录 J 的最终 loss/PPL、step、累计训练目标 token、训练 token/s、峰值显存、阶段与总
耗时、停训原因。仅完整验证成功、源码未变、累计不超过六小时的结果进入有效比较。
报告检查 J 的完整验证 PPL 是否低于 H 的精确值 36.14949117609303；无效结果不宣称优势。

历史 H 为 1,800,273,920 训练目标 token、86,654.24 token/s、总耗时 21,173.87 秒。
单卡负载可能改变吞吐和训练量，历史 H/J 比较评价整套六小时运行结果，不能将全部差异
归因于 LR；相近累计 token 的区间训练 loss 仅为辅助证据。即使更好也只列为新候选，
不自动更新 dev，不把单次结果当作统计显著结论。

本地仅小规模测试：

```bash
python -m pytest tests/test_peak_lr_single_suite.py tests/test_context_lr_experiments.py \
  tests/test_best_lr_reference_suite.py tests/test_optimizer_batch_suite.py tests/test_training.py \
  tests/test_optimizer.py tests/test_tokenizer.py tests/test_train_bpe.py tests/test_parallel_bpe_experiment.py -q
```

## 任务一：启动并发化 × 解码优化（不重新训练）

新增 `benchmark_task1_pipeline.py` 与实验专用 `task1_pipeline_runtime.py`，仅在
`exploration` 探索，不改 `dev`、`bpe.py`、课程模板 `train_bpe.py` 或 Transformer 入口。
**只加载 `outputs/8b68b68e0cbe/tokenizer`，不调用 BPE.train()。** 旧传输工具的正常测速
路径不变；辅助 worker 仅在诊断时增加初始化接收、BPE 重建及就绪时间记录。

实验启动时读取上一轮 `manifest.json` / `selection.json`，确认推荐 D、原 16 workers /
256 分片 / 1M 缓存 / 128K 解码块配置，并比对完整验证原文及三个 tokenizer JSON 的
SHA-256。旧文件可位于不同绝对路径，按旧清单的文件名和指纹识别；不允许静默换词表。
历史约 **15.622 秒**只作参考，最终推荐依据本轮新测 D0，不能拿历史成绩直接排名。

第一阶段为四种完整解码专项，每种 3 次，新进程、固定种子交错执行：

| 解码 | 方法 |
|---|---|
| C0 | 当前字典查找 / map / 分块 join |
| C1 | 稠密查找表 / map / 分块 join |
| C2 | 字典 / 标准库 operator.itemgetter 批量查找 |
| C3 | 稠密查找表 / operator.itemgetter 批量查找 |

查找表、索引及临时对象均在解码计时内构造；所有方案拼接原始 bytes 后才做一次 UTF-8
解码（errors=replace），保留空、单 ID、稀疏词表和非法 ID 行为，不拼接单 token 解码文本。
解码块固定 131,072 IDs。从 C1–C3 中选解码中位数最快者 Q，进入完整接口消融；即使 Q
在专项慢于 C0，仍进入完整消融检查实际 encode 产物上的效果，不直接据专项推广。
**12 次解码专项不属于任务一完整成绩。** 装载参考 IDs 在专项计时之外，不预解码文本。

第二阶段使用以下四种组合，每种 5 次：

| 方案 | 初始化 | 解码 |
|---|---|---|
| D0 | 当前 D：启动一个 worker 后同步发送状态 | C0 |
| D1 | 先创建全部 worker，再发送初始化状态 | C0 |
| D2 | 当前 D | Q |
| D3 | 先创建全部 worker，再发送初始化状态 | Q |

固定 16 workers、256 个安全分片、每 worker 1M 满后清空缓存。全部 worker 就绪后，
沿用 D 的动态任务分发与按编号组装结果。不会直接 fork 已启动 W&B 的多线程进程。

每次完整正常计时为 **BPE.encode → 完整 list[int] → BPE.decode → 原文比较**，包括
分片、启动、状态传输与重建、编码、结果传输、列表组装、退出及清理。使用刚加载的 BPE，
空缓存、冷 worker；不在计时前启动进程或准备完整编码/解码结果。只预先加载分词器和
读取原文。每次子试验模拟课程调用方导入 W&B，但不单独创建 run。

20 次筛选采用固定种子交错顺序，最快候选与 D0 再做 5 对独立复测，每对随机先后。
至少降低总耗时中位数 **3%** 且赢得至少 **4/5 对**才推荐替换，否则保留 D0。
D0 本身最快则跳过自身配对。最终推荐另做一次独立完整确认，不挑最快单次。
最多 31 次完整正常测速，另有 12 次解码专项、最多 2 次独立诊断和一次可选参考编码。

所有完整候选都必须通过 **66,401,098 个 ID** 的完整精确对照和全文还原，任何错误立即
停止。默认串行核心独立生成一次临时 uint32 little-endian 参考，**仅编码，不训练**；
若已有这套词表对应的可信完整参考文件，可传 `--reference-tokens` 直接复用（应为
265,604,392 bytes）。公共数据只读，临时参考位于个人 outputs，完成或失败后清理。
仅 token 数一样不足以认定参考正确，最终仍精确对照每一个 ID。

独立诊断记录每个进程创建、初始化序列化/发送、接收与重建、就绪、首个结果、编码与打包、
等待、组装、退出、解码建表/索引/查找/拼接/UTF-8、CPU affinity 和 RSS。为拆分查找与
拼接，解码诊断物化临时元组，分配模式与正常测速不同，不能排名。跨进程时间重叠，
不能相加；发送/接收包含序列化、阻塞或等待。worker 峰值之和不是同时发生的实际峰值。

云端整个探索串行使用相同的前 16 个允许逻辑 CPU，也可指定恰好 16 个 `--cpu-ids`；
不足直接报错，不降配。父进程与后台记录服务受 affinity 限制；worker 各绑定一个 CPU，
线程变量为 1，GPU 隐藏。worker 异常、中断或初始化部分失败均终止并回收所属进程，
不返回部分列表。只创建一个 `bpe-task1-pipeline-exploration` W&B run，保存完整源码、
配置、候选 Table 和结果。个人输出保存指纹、实际顺序、逐次 JSON、CSV、报告和选优；
原文、词表、历史清单、参考或本轮源码发生变化时不产生有效推荐。

云端命令（先配置个人 W&B 环境变量，输出目录须不存在）：

```bash
source /env/activate_a1.sh

python benchmark_task1_pipeline.py \
  --tokenizer-prefix outputs/8b68b68e0cbe/tokenizer \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --previous-results-dir outputs/task1-transport-exploration \
  --decoder-repeats 3 \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-pipeline-exploration
```

`--previous-results-dir` 指向云端已有上一轮清单及选优报告的目录；如果历史文件移到了
其他目录，修改此路径即可，不需要重新跑旧实验或重新训练。只有输入指纹完全一致才继续。
实验只报告推荐，不自动更新 dev，不承诺达到 5 秒，也不替代课程入口正式 W&B 成绩。

本地仅小文本检查（开发模式使用 2 workers，禁止用于正式成绩）：

```bash
python -m pytest tests/test_task1_pipeline.py tests/test_task1_transport.py \
  tests/test_task1_efficiency.py tests/test_tokenizer.py tests/test_train_bpe.py -q
```

## 任务一：批量／并行解码与普通 BPE 接入验收

新增 `benchmark_task1_decode.py`，只在 `exploration` 探索。继续加载
`outputs/8b68b68e0cbe/tokenizer`，**不重新训练、不修改项目的 bpe.py、dev 或课程模板**。
上一轮 D3/C2 确认约 11.167 秒；本轮锁定其并发启动、16 编码 workers、256 安全分片、
每 worker 1M 清空缓存及全部合并规则。历史成绩不参与新排名，重新测 R0 基线。

第一阶段比较六种解码，各 3 次，固定种子交错、新进程运行：

| 方法 | 实现 |
|---|---|
| R0 | 上一轮 C2，131,072 IDs 一块 |
| N128 | NumPy object 词表，批量索引；每块 131,072 IDs |
| N1M | 同上，每块 1,048,576 IDs |
| P4 / P8 / P16 | 新建 4 / 8 / 16 个独立轻量解码 worker |

NumPy 为项目已有依赖，不安装新依赖或替代 Tokenizer。使用 object 保存原始 bytes，
保留零字节，不使用会改变尾部零字节的定长字符串表示。建表、ID 转换与合法性检查、
索引、tolist、拼接全部计时；禁止浮点截断、负索引和整数溢出改变 ID。非连续词表及
小输入保留 C2 的轻量行为。计时前不人为预热 NumPy，记录它是否已由调用方加载及
计时期间新增模块；首次导入发生在计时内时也计入。

并行解码每任务固定 1,048,576 IDs，worker 内按 131,072 IDs 查找、拼接并返回 bytes。
worker 只用标准库，不导入 BPE、merges、W&B、模型或 NumPy。按编号汇总后才做统一
UTF-8 解码（errors=replace），不拼接逐 token/分块解码字符串。打包、创建、初始化、
IPC、组装、退出和清理全部在 decode 内；不复用编码进程池或 encode 保存的原文、ID
副本、还原结果缓存。decode 仅依据其 IDs 与当前词表工作。

这 **18 次解码专项不是完整成绩**。各选最快 NumPy 和最快并行方法进入第二阶段，
与 R0 共三组；编码全部固定 D3。每组 5 次完整
**encode → 完整 list[int] → decode → 原文比较**，再将最快候选与 R0 做 5 对独立复测。
至少降低中位数 3% 且至少 4/5 对胜出才推荐；R0 最快则跳过自身配对。最终推荐再做
一次独立完整确认。最多 26 次这一阶段的完整正常测速，诊断另外进行、不能排名。
当前编码本身约 6.34 秒，本轮不承诺进入总耗时 5 秒。

第三阶段在本次个人 outputs 下生成 `plain-source` 独立源码快照，固化推荐私有后端：
普通 `from bpe import BPE`、load/encode/decode 不执行 selected_backend/selected_pipeline
替换，也不增加协调层。encode 的启动使用同一并发启动实现；小输入保留轻量路径，
不为前 200 token 展示启动 worker。BPE 训练、合并、预分词、构造与序列化不变，
encode_iterable 继续 2K 缓存。该快照不覆盖项目源文件，并验证课程模板逐字节不变。

普通路径与实验推荐路径再做 5 对交错复测。普通路径中位数须≤实验路径 103%，并至少
4/5 对≤对应实验路径 105%。不满足则 `ready_for_plain_confirmation=false`，记录回退而
不宣称接入完成。生成的候选源码与来源哈希完整上传单独 source-code artifact，包含
未修改的课程源码记录器。运行过程中和后续确认都核对同一份快照，禁止临时修改。

云端探索命令（预先配置个人 W&B 环境，输出目录须不存在）：

```bash
source /env/activate_a1.sh

python benchmark_task1_decode.py \
  --phase exploration \
  --tokenizer-prefix outputs/8b68b68e0cbe/tokenizer \
  --valid-file /data/assignment1-data/owt_valid.txt \
  --previous-results-dir outputs/task1-pipeline-exploration \
  --decoder-repeats 3 \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-decode-exploration
```

上一轮目录须含 manifest.json / selection.json，且推荐已完成的 D3/C2。按原文及三个
分词器文件的 SHA-256 校验，可移动文件路径但不能换词表或原文。全部候选必须逐项
对照 **66,401,098 IDs** 并还原全文，任何正确性失败立即停止。支持 `--reference-tokens`
复用该词表对应的可信 uint32 little-endian 完整 ID 文件；否则只用串行核心生成一次参考，
不训练。默认保留本次 outputs/reference.u32le（约 265.6 MB）供独立确认，第二阶段前
不要删除该文件、探索目录、候选快照或原文/词表。提供外部参考时也须保留其原路径。

接入门槛通过后，单独执行活跃 W&B 上下文确认：

```bash
python benchmark_task1_decode.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-decode-exploration/selection.json \
  --output-dir outputs/task1-plain-confirmation
```

确认继承并核对第一阶段实际输入路径、参考、候选源码及同一组 CPU；仍只加载旧词表。
独立普通验证进程先建立真实活跃的 W&B run，再导入候选 BPE、加载词表和读取原文，
计时只涵盖完整公开 encode/list/decode/比较，首次运行模块导入仍照常计入。记录实际
BPE 来源、计时前模块、计时中新导入模块、活跃状态、RSS 和完整正确性，不用实验替换。
第二个 run 的 job_type 为 `bpe-task1-plain-confirmation`，课程记录器直接保存完整候选
源码，并展示前 200 token；探索 run 为 `bpe-task1-decode-exploration`。仅这两个独立 run，
不为每个测速子进程建立 run。确认比接入中位数慢超过 5% 时标记上下文回退，单次结果
不作稳定性证明，不自动修改 dev，也不冒充要求重新训练的课程入口记录。

全程 GPU 隐藏，默认固定当前允许的前 16 个逻辑 CPU；可给恰好 16 个 `--cpu-ids`。
编码与解码 worker 不重叠，每个 worker 绑定一个 CPU、内部线程 1，不超 16 核。
不允许资源不足静默降配或按失败结果选优；部分启动失败、worker 异常和中断均回收
所属进程，不返回部分列表。公共数据只读，临时产物和源码快照均在个人 outputs。
分阶段报告区分解码专项、完整接口、接入配对和活跃确认；独立诊断记录建表、转换、
打包、启动、查找、IPC、拼接、UTF-8 和内存，重叠时间不能相加，RSS 独立峰值之和
不是同时发生的真实峰值。源码、原文、词表、历史清单或参考改变则结果无效。

本地只做小文本测试；开发模式限制为 2 workers，并标记不可当作正式资源/成绩。
小文本活跃 W&B 测试采用 offline 模式，不上传：

```bash
python -m pytest tests/test_task1_decode.py tests/test_task1_pipeline.py \
  tests/test_task1_transport.py tests/test_task1_efficiency.py \
  tests/test_tokenizer.py tests/test_train_bpe.py -q
```

### 任务一：T0–T3 编码传输探索（不重新训练）

`benchmark_task1_encode_transport.py` 从前一轮 `selection.json` 和活跃 W&B 的
`confirmation.json` 继承并核对分词器、完整验证原文、参考 ID、CPU 集合及源码快照。
T0 使用验收通过的普通 R0 快照，不使用项目当前默认 BPE 代替历史基线；
T1 在每次 encode 内只序列化一次公共词表状态，T2 增加每 worker 共享输出槽位，
T3 再增加共享 UTF-8 输入。四项都使用直接轻量 worker、并发启动及 C2 解码，
没有协调进程、预热进程池或新的编译依赖。

```bash
source /env/activate_a1.sh

python benchmark_task1_encode_transport.py \
  --previous-selection outputs/task1-decode-exploration/selection.json \
  --plain-confirmation outputs/task1-plain-confirmation/confirmation.json \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-encode-transport
```

前阶段报告内的云端绝对路径及源码快照必须仍存在；本地下载的结果目录不能直接替代
云端源码快照。输出目录须不存在。默认继承相同的 16 个逻辑 CPU，`--cpu-ids` 只能
显式重申同一集合，不能更换资源。各 worker 绑定一个 CPU，内部线程为 1，GPU 隐藏。
T2/T3 需要 Python 3.13+；计时内检查 `/dev/shm` 的实际需求和 64 MiB 余量，
不足时标记不可用并排除该候选，不回退成管道实现。

每次使用新子进程、重新加载词表及空缓存；完整计时公开 encode、完整 `list[int]`、
decode 和原文比较，包括共享内存准备、状态序列化、进程启动、传输、列表组装、
退出与清理。参考 ID 的完整精确比较在计时外执行，正常验证必须为 66,401,098 个 ID。
共享输出被复制为父进程独立持有的数组后才允许 worker 复用槽位；解码不读取原文缓存。
异常或中断会回收所属 worker 并释放共享内存，不返回部分列表。

流程为基线诊断、四项各五次交错筛选、最快新候选诊断、与基线五对独立复测及一次确认。
中位数降低至少 3% 且至少胜出 4/5 对才推荐替换，否则保留 T0。随后自动生成普通 BPE
源码快照并做五对接入验收：普通路径中位数不超过实验路径的 103%，且至少 4/5 对不超过
对应实验路径的 105%。诊断不参与排名，跨进程阶段不能相加解释总时间；worker 峰值 RSS
之和也不代表同时发生的实际峰值。

接入门槛通过后，单独执行活跃 W&B 确认（不再次启动探索 run）：

```bash
python benchmark_task1_encode_transport.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-encode-transport/selection.json \
  --output-dir outputs/task1-encode-transport-plain-confirmation
```

探索创建一个 `bpe-task1-encode-transport-exploration` W&B run，确认再创建一个
`bpe-task1-plain-confirmation` run；记录完整源码、配置、候选 Table、JSON/CSV/Markdown
及指纹。始终不训练、不修改 `train_bpe.py`、默认 BPE 或 `dev`。加载旧词表的探索及
确认不替代课程训练入口成绩，也不预先承诺低于 5 秒。

本地小文本检查：

```bash
uv run pytest -q tests/test_task1_encode_transport.py
```

`--development --no-wandb --repeats 1` 仅用于匹配 development 前阶段报告的小文本流程，
不能将其产物当作正式成绩。

### 任务一：Cython 解码、未命中编码热点探索

`benchmark_task1_native.py` 在 `exploration` 比较 B0、CY-D、CY-ED，不训练词表，
不改变默认 BPE、序列化格式或课程入口。B0 来自先前验收通过的 **R0 源码快照**，
不是报告被 W&B 上传异常标为未完成的 T0–T3 运行。

| 候选 | 编码 | 解码 |
|---|---|---|
| B0 | R0 直接 worker、并发启动 | C2 |
| CY-D | 与 B0 相同 | 原生双遍批量解码 |
| CY-ED | 原生未命中合并核心，保留缓存和并行框架 | 原生双遍批量解码 |

依赖只安装到个人 conda base；继续使用你现有的 base 与课程环境组合。
不必为本实验更换项目构建后端，或重新安装 PyTorch、NumPy、Tokenizer：

```bash
conda install -n base -c conda-forge \
  "cython>=3.1,<4" setuptools wheel c-compiler

conda activate base
source /env/activate_a1.sh

python -c "import sys, Cython, setuptools; print(sys.executable); print(sys.version); print('Cython:', Cython.__version__)"
```

随后始终使用上述同一个 `python` 编译和运行。要求 CPython 3.12/3.13；环境组合后
若无法导入 Cython，先修正激活顺序，不要用另一个 Python 编译来绕过问题。

```bash
python build_task1_native.py \
  --output-dir outputs/task1-native-build

python benchmark_task1_native.py \
  --previous-selection outputs/task1-decode-exploration/selection.json \
  --plain-confirmation outputs/task1-plain-confirmation/confirmation.json \
  --build-dir outputs/task1-native-build \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-native
```

构建和实验输出目录均须不存在。前阶段报告引用的云端原文、分词器、参考 ID 和源码
快照须保留；默认继承同一组 16 核和 16 workers、256 安全分片、每 worker 1M
清空缓存。线程数为 1、GPU 隐藏。扩展只在显式构建命令中编译；实验不自动安装、
编译或因扩展缺失而静默回退。修改 `.pyx` 后须使用新目录重新构建。

每轮先对 B0/CY-D 各做五次交错测速及五对独立复测。CY-D 完整耗时中位数降低至少
3%、且赢得至少 4/5 对时，才对 CY-D/CY-ED 执行同样流程。否则不测 CY-ED。
推荐方案独立确认后，再做五对普通接口接入验收：中位数不超过实验路径 103%，
且至少 4/5 对不超过对应实验路径 105%。诊断另跑，不参与排名。

计时包含公开 encode、完整 `list[int]`、decode、原文比较，以及模块首次加载、
原生查找表建立、运行分配、进程启动/传输/退出/清理。文件读取、词表加载、预编译
和参考核对在计时外。每次完整核对 66,401,098 个 ID；失败立即停止。
原生解码单线程持有 GIL，直接检查 list ID、累计长度、一次分配并 memcpy，最后
整体 UTF-8 replace 解码；不读取 encode 原文。稀疏词表或非普通整数走兼容路径，
但完整候选测速明确检查实际 native 后端。编码使用整数符号和 C 哈希 rank 表，
沿用扫描最低 rank、批量合并非重叠 pair 的原算法；不引入堆或链表。

普通接口验收通过后，单独执行活跃 W&B 确认：

```bash
python benchmark_task1_native.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-native/selection.json \
  --output-dir outputs/task1-native-plain-confirmation
```

需要保留构建目录和候选源码快照。一个 `bpe-task1-native-exploration` W&B run
覆盖探索；确认另建 `bpe-task1-plain-confirmation` run。记录 `.pyx`/`.pxd`、生成 C、
构建脚本、扩展二进制及 SHA-256，Python ABI、Cython、编译器和编译参数；
二进制上传为文件，不当作 UTF-8 源码打印。普通接口核对实际扩展路径及 ABI。
上传异常写 `upload_failure.json`、`upload_status=failed`，与计算失败分开；
`submitted` 表示已交给 W&B 客户端，并非远端文件独立验收。原始历史报告不作修改。

本地只做小规模测试，不以小文本速度判断提升；完整成绩留在云端：

```bash
python -m pytest -q tests/test_task1_native.py
```

加载旧词表的探索/确认不替代课程训练入口成绩，不自动更新 `dev`，也不预设能达到 5 秒。

### 任务一：原生编码加载适配层复测（复用已编译扩展）

`benchmark_task1_native_adapter.py` 继承上一轮完成的 CY-D 结果和活跃 W&B 确认，
复用 CY-D、CY-ED 源码快照中的二进制，不安装依赖、不编译、不训练。A0 为原 CY-D；
A1 从 CY-ED 派生，**仅替换私有 `load_native()`**：首次完成 ABI、二进制及实际加载
路径核验后才缓存模块，后续直接返回，不再重复解析路径或读取元数据。原合并算法、
缓存、统计、进程传输及解码保持不变。默认 BPE、旧探索工具和 `dev` 不变。

固定解释器，避免课程环境抢占 PATH：

```bash
bpe_python="/home/zhangziyang/miniforge3/envs/bpe-native/bin/python"

"$bpe_python" benchmark_task1_native_adapter.py \
  --previous-selection outputs/task1-native/selection.json \
  --plain-confirmation outputs/task1-native-plain-confirmation/confirmation.json \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-native-adapter
```

输出目录须不存在；前阶段原文、词表、参考、原构建文件和 CY-D/CY-ED 候选快照须保留。
预检确认各组 `.pyx`、生成 C、二进制及 ABI 一致，继承同一组 16 CPU、16 workers、
256 安全分片、每 worker 1M 清空缓存；线程数为 1，GPU 隐藏。不要使用裸 `python`
或 `uv run` 来切换解释器；没有 `--build-dir` 或自动重编译入口。

`--diagnostics` 对 A0、原 CY-ED、A1 各独立诊断一次。诊断使用独立源码快照，计数
加载调用、ABI/二进制验证调用、实际路径解析及对应耗时，同时保留 worker 原生表
构建、编码、传输、列表组装和解码数据。实际路径解析和验证耗时包含在加载耗时中，
不能相加；诊断额外计数不进入排名。正常快照不安装诊断包装器或计数分支。

正常流程为 A0/A1 各五次交错筛选、五对独立随机顺序复测、一次推荐方案确认，再做
五对普通接口接入验收，共 31 次正常测速。原 CY-ED 只作诊断，不进入排名。
推广要求完整中位数至少降低 3% 且胜出至少 4/5 对；否则保留 A0。接入要求普通
路径中位数不超过实验路径 103%，且至少 4/5 对不超过对应实验路径 105%。每次
使用新试验进程、空缓存，完整精确核对 66,401,098 个 ID 及全文还原；错误立即停止。
首次扩展加载、状态准备、进程启动、传输、组装和清理均计入 encode/list/decode/比较。

接入通过后，使用相同解释器独立确认：

```bash
"$bpe_python" benchmark_task1_native_adapter.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-native-adapter/selection.json \
  --output-dir outputs/task1-native-adapter-plain-confirmation
```

报告以 A0/A1 标记方案，原生后端元数据仍保留 CY-D/CY-ED；独立确认增加
`adapter_case` 标记。活跃 W&B 超过接入中位数 5% 时保留回退标记；上一轮约 12.95%
的上下文回退也原样写入历史说明，不将它当成 ID 错误，不直接归因于 W&B。
逐次 JSON/CSV/Table 记录前后 UTC 时间及整机 load average（不是本组 CPU 利用率）。
一个 `bpe-task1-native-adapter-exploration` run 记录探索全部源码、候选和诊断快照；
独立确认另建一个 `bpe-task1-plain-confirmation` run。上传异常与计算有效性分开记录。
试验前后和上传结束后检查指纹；不自动更新 `dev`，不预先承诺低于 5 秒。

本地仅执行小文本回归：

```bash
.venv/bin/python -m pytest -q tests/test_task1_native_adapter.py
```

测试会在临时目录构建小文本所需的本机扩展以模拟前阶段快照；云端探索入口本身
只复用既有扩展，不运行编译器。

### 任务一：列表组装与整数池复用探索

`benchmark_task1_assembly.py` 从上一轮通过验收的正常 A0/CY-D、A1/CY-ED 快照派生候选。
先固定 CY-D 比较父进程列表组装；胜出后再与修复后的 A1 编码组合。`dev`、默认 BPE、
课程入口和原生解码二进制不变，不重新训练词表，也不扫描分片数。

| 方法 | 父进程组装 |
|---|---|
| M0 | 原 `list.extend(array('I'))` |
| M1 | Cython 正常创建/取得 Python 整数，构造分片列表再追加 |
| M2 | 与 M1 相同，但从本次 encode 创建的整数池复用整数对象 |

M1/M2 使用相同的临时列表与最终列表增长方式。输入是实际 worker 返回的连续 uint32
buffer；检查格式、4 字节大小、对齐、ID 范围及长度溢出。整数池按词表建立，不使用
参考 token 数预分配结果。对新列表采用正确的引用转移，最终返回真实完整 `list[int]`；
池释放后列表仍独立有效。原生解码照常根据传入 ID 和词表还原文本，不读取原文缓存。

新增的是独立组装扩展；已有 Cython/编译器即可，不需要增加依赖。仍须使用固定解释器：

```bash
bpe_python="/home/zhangziyang/miniforge3/envs/bpe-native/bin/python"

"$bpe_python" build_task1_assembly.py \
  --output-dir outputs/task1-assembly-build

"$bpe_python" benchmark_task1_assembly.py \
  --previous-selection outputs/task1-native-adapter/selection.json \
  --plain-confirmation outputs/task1-native-adapter-plain-confirmation/confirmation.json \
  --build-dir outputs/task1-assembly-build \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-assembly
```

构建和实验输出目录须不存在；上轮正常 A0/A1 快照、原构建产物、词表、原文及参考文件
必须保留。预检拒绝诊断快照和 ABI/指纹不匹配，复制原解码扩展时保持其字节完全不变。
只显式构建新扩展，测速期间不自动编译、不静默回退。修改组装 `.pyx` 后需显式重新构建。

资源固定为相同 16 个逻辑 CPU、16 workers、256 安全分片、每 worker 1M 清空缓存；
内部线程为 1，GPU 隐藏。计时覆盖公开 encode、完整 list、decode 和原文比较，包括
新扩展首次导入、整数池准备、临时列表和引用处理、进程启动、传输、退出及清理。
文件读取/词表加载和完整参考核对在计时外；每次须匹配 66,401,098 个 ID 并全文还原。

流程：

1. M0/M1/M2 各五次固定种子交错测速；若 M0 最快则保留它，否则最快新方案与 M0
   做五对独立随机顺序复测。完整中位数降低至少 3% 且胜出至少 4/5 对才推广。
2. 若推广 M2，再做 M2/M1 五对机制比较，判断整数池的独立收益；这一结果只作解释，
   不改变已经通过主对照的方案。
3. 若推广 M1/M2，固定其组装方式，对 CY-D 与修复后的 A1 编码各做五次交错测速、
   五对独立复测；用相同门槛判断是否组合。未推广组装则跳过组合阶段。
4. 最终方案独立确认一次，再做五对普通接口验收：中位数≤实验路径103%，且至少
   4/5对≤对应实验路径105%。只保存最终推荐，不挑选最快单次作为成绩。

`--diagnostics` 对三组使用单独快照，解码改用辅助扩展中的诊断副本，复现原解码算法并
分别计时词表准备、ID检查/长度累计、bytes分配、复制及首次触页、UTF-8解码与清理。
诊断副本不参与排名，不改变正常解码；不额外清零输出缓冲区，也不把 allocation 时间
解释为已支付全部首次写入成本。组合阶段另做一轮组合诊断。诊断、重叠阶段及worker
峰值RSS之和不能相加当作完整耗时或实际同时峰值。记录父进程峰值、worker内存、任务
分布、每次试验前后UTC及整机load average。

普通接口验收通过后：

```bash
"$bpe_python" benchmark_task1_assembly.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-assembly/selection.json \
  --output-dir outputs/task1-assembly-plain-confirmation
```

探索为一个 `bpe-task1-assembly-exploration` W&B run，活跃确认另建一个
`bpe-task1-plain-confirmation` run；活跃确认超过接入中位数5%时标记回退。W&B和本地
JSON/CSV/Markdown保留全部候选源码、`.pyx`、生成C、新旧二进制、ABI、编译器、编译参数
及SHA-256；普通确认记录 `assembly_case` 和实际加载路径。上传失败与计算失败分开记录，
运行前后及上传结束后核对指纹。不自动更新 `dev`，不预先承诺低于5秒。

本地只做小文本回归（测试会在临时目录编译本机扩展，云端实验入口本身不编译）：

```bash
.venv/bin/python -m pytest -q tests/test_task1_assembly.py
```

### 任务一：解码对象访问与分片数探索（不更新 dev）

下一轮从已通过普通接口接入的正常 **M2-A1** 快照派生，复用原词表和参考 ID，不训练。
第一阶段固定 256 分片，比较 N0（原原生解码）、N1（持有 GIL、借用列表元素引用）、N2（N1 加调用内 uint32 索引）。
N2 的索引按实际输入列表长度分配，完整验证约额外 253 MiB；准备和释放均计时，不使用 encode 留下的数组。
第二阶段固定胜出的解码，比较 64/128/256/512 个安全分片；仍为 16 workers、每 worker 1M 清空缓存及整数池组装。

每组五次交错筛选，最快新方案与阶段基线再做五对复测：完整耗时中位数至少降低 3%，且至少胜出 4/5 对才推广。
N2 被推广时另做 N1/N2 五对机制对照。随后独立确认、五对普通接口接入验收；活跃 W&B 确认单独执行。
全部排名计时为公开 encode → 完整 list[int] → 公开 decode → 原文比较，首次扩展导入、进程退出及缓冲清理均计入。
每次核对完整 66,401,098 个参考 ID 并还原全文；诊断不进入排名。不足以生成要求数量非空安全分片的配置明确排除，不静默串行回退。

只支持带 GIL 的 CPython 3.13，使用已有个人环境，无新增依赖。原编码、整数池扩展不重编译；仅显式编译独立解码扩展：

```bash
bpe_python="/home/zhangziyang/miniforge3/envs/bpe-native/bin/python"

"$bpe_python" build_task1_decode_access.py \
  --output-dir outputs/task1-decode-access-build

"$bpe_python" benchmark_task1_decode_access.py \
  --previous-selection outputs/task1-assembly/selection.json \
  --plain-confirmation outputs/task1-assembly-plain-confirmation/confirmation.json \
  --build-dir outputs/task1-decode-access-build \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-decode-access
```

保留上轮原始快照和构建产物；新输出目录须不存在。预检核对源码、ABI、二进制及全部输入指纹，并继承相同 16 核硬限制，GPU 隐藏、内部线程为 1。
只读分片预检使用单独解释器；正常计时仍重新计算分片，不复用预检结果或编码缓存。
结果包含阶段选择、分片数、RSS、诊断分项、系统负载和历史上下文回退标记。
N2 的诊断检查/累计时间包含索引写入，复制时间包含首次触页；不将重叠阶段相加。
上传失败与计算有效性分别记录，不将单次探索结果作为课程正式成绩。

接入门槛通过后，以同一解释器执行活跃 W&B 独立确认：

```bash
"$bpe_python" benchmark_task1_decode_access.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-decode-access/selection.json \
  --output-dir outputs/task1-decode-access-plain-confirmation
```

普通接口中位数须不超过实验路径的 103%，至少 4/5 对不超过对应实验路径的 105%。
活跃确认比普通接口中位数慢超过 5% 时保留回退标记，不直接归因于 W&B。
本地只编译与验证小文本；完整验证实验在云端执行，不修改默认 BPE 或课程入口。

### 任务一：父进程调度与最终列表组装探索

第 5 轮保留 **N2 + 256 分片**：N0/N2 配对中位数 7.183/6.120 秒，降低 14.79%，5/5 胜出。
N2 对 N1 的额外收益为 3.35%，4/5 胜出，代价是约 253 MiB 的调用内索引。
128 分片复测未过门槛（6.012/6.054 秒，仅胜出2/5），不推广。
普通接口中位数 6.003 秒，活跃 W&B 确认 6.080 秒（+1.28%）；历史上下文回退标记不改写。
活跃确认中 encode/decode/比较分别约4.019/1.875/0.187秒，因此下一轮先探索父进程编码流水。

上轮探索的计算及正确性有效，但 W&B Table 上传失败：N0/N1/N2 诊断行的字典字段不同，触发列类型冲突。
现已将动态字典和数组列统一序列化为 JSON 文本，保留稳定标量列；逐次 JSON 仍为完整结构。
修复同时用于旧解码访问入口及新入口；不修改历史报告，也不将历史上传失败改为成功。

| 候选 | 相对本轮 N2/256 的变化 |
|---|---|
| Q0 | 已通过普通接口验收的本轮路径 |
| Q1 | 收到结果后先派发下一任务，再组装当前结果 |
| Q2 | Q1 + Cython 直接将池中整数追加到最终 list，取消分片临时 list |
| Q3 | Q1 + 保留实际返回数组，收齐并退出 worker 后按真实长度一次分配完整 list |

四组固定16 workers、256安全分片、每worker 1M清空缓存；原编码、整数池及N2解码二进制保留。
Q2/Q3使用独立扩展；Q3仅依据实际收到的数组长度分配，禁止使用参考token数量预分配。
Q3包含暂存原始uint32数组的内存成本，并可能损失原有流水重叠，不预设胜出。
默认四组各5次交错筛选，最快新组再与Q0做5对复测，要求中位数至少降低3%、至少赢4/5对。
若Q2/Q3推广，再与Q1做5对机制对照；该对照不改变主要推广结论。
随后独立确认和5对普通接口接入，活跃W&B确认单独运行。
全部计时仍是公开encode→完整list[int]→公开decode→原文比较，所有退出、暂存、最终组装及清理均计入。
诊断单独记录finish_assembly_seconds，避免漏计Q3收尾组装；跨进程重叠阶段不相加。
每次精确对照完整66,401,098个ID并还原全文；错误停止，候选串行，CPU硬限制继承同一16核。

使用固定个人解释器，无需新增依赖或重训，仅编译新的父进程组装扩展：

```bash
bpe_python="/home/zhangziyang/miniforge3/envs/bpe-native/bin/python"

"$bpe_python" build_task1_parent_assembly.py \
  --output-dir outputs/task1-parent-assembly-build

"$bpe_python" benchmark_task1_parent_pipeline.py \
  --previous-selection outputs/task1-decode-access/selection.json \
  --plain-confirmation outputs/task1-decode-access-plain-confirmation/confirmation.json \
  --build-dir outputs/task1-parent-assembly-build \
  --repeats 5 \
  --diagnostics \
  --output-dir outputs/task1-parent-pipeline
```

接入验收通过后：

```bash
"$bpe_python" benchmark_task1_parent_pipeline.py \
  --phase plain-confirmation \
  --selection-report outputs/task1-parent-pipeline/selection.json \
  --output-dir outputs/task1-parent-pipeline-plain-confirmation
```

保留前阶段正常N2快照、原构建及参考文件，新输出目录须不存在。
源码、三个原扩展、新扩展及输入均做指纹核验，记录Python ABI、编译器及实际加载路径。
接入中位数不超过实验路径103%，至少4/5对不超过对应实验路径105%；活跃确认超出接入中位数5%仍保留标记。
仅在exploration探索，不自动更新dev、默认BPE或课程入口；不承诺低于5秒。
