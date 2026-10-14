# BPE 与 Transformer 作业

正式源码包括 `bpe.py`、`train_bpe.py`、`transformer_module.py`、
`transformer_main.py`、`parallel_bpe_encoding.py` 和课程提供的 `wandb_record.py`。
模型核心、BPE 和课程源码记录器保持不变。

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
普通预分词片段。正式入口的 W&B 配置记录 `clear_all` 与实际缓存容量。
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
