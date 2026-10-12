# BPE 与 Transformer 作业

保留五个核心文件：`bpe.py`、`train_bpe.py`、`transformer_module.py`、
`transformer_main.py` 和课程提供的 `wandb_record.py`。
模型核心、BPE 和课程源码记录器保持不变。

## 环境与运行

```sh
uv python pin 3.13
uv sync --locked
uv run wandb login
export WANDB_ENTITY="你的实际账号或团队"
export WANDB_PROJECT="lmfs-assignment1"
export STUDENT_ID="你的学号"
export WANDB_MODE="online"
export PYTHONIOENCODING="utf-8"
```

任务一入口：

```sh
uv run python train_bpe.py
```

任务二只修改 `transformer_main.py` 顶部的 `CONFIG` 字典，然后在项目根目录执行：

```sh
uv run python transformer_main.py
```

不再使用命令行参数、开发清单、梯度累积或自动验证耗时估算。
配置中的原始训练和验证文件必须是完整 OWT 文件。分词器默认指向本人已训练的
`outputs/830f59c92e15/tokenizer`；迁移时携带同前缀的 vocab、merges、config 三个 JSON 文件。

模型配置为 4 层、512 维、8 头、FFN 1408、上下文 512、BF16。
batch 64，即每次更新 32,768 token，面向 A100/A800 80GB，尚未在该卡上确认吞吐。
AdamW β=(0.9,0.999)、权重衰减 0.1、梯度裁剪 1.0。
学习率为 3e-4→3e-5，warmup 200 步，余弦终点 20,000 步；之后保持最低学习率。
这些是候选超参数，不是已验证的最优值。更改学习率或路径直接修改 `CONFIG`。

## 课程要求保留项

- 使用本人 BPE 编码完整原文，不额外插入标记；模型从随机权重开始。
- 编码后写入磁盘 token 数组，从整个训练数组随机抽取连续窗口。
- 模型各层、注意力、交叉熵、AdamW、调度和梯度裁剪均为手写实现。
- 每 100 次更新向 W&B 记录区间训练 PPL，尾部不足 100 步也记录；日志损失在设备上累计。
- 仅保存最后模型及优化器、步数和随机状态到本地 `last.pt`，随后验证完整 OWT 验证集。
- 验证关闭梯度，按总目标 NLL / 总目标数计算 PPL，尾部计分，仅首 token 无预测目标。
- 从入口最顶部计时，导入、W&B、源码、编码、初始化、训练、保存和最终验证均计入累计六小时。
- 固定预留 1,200 秒收尾，不再自动估算；须在目标设备上确认足够，超时会报错。
- W&B 记录源码、配置、最终 PPL、步数、UTC 起止时间、累计耗时及是否超出六小时。

本地不再额外生成 `metrics.jsonl`、`config.json`、`result.json`、`timing.json` 或数据指纹文件。
课程记录器仍会保存、上传源码快照；这部分是明确的课程要求。
已有分词器、数据缓存、模型和历史运行记录不删除。

续训时，在同一个配置字典中填写 `resume`、`resume_from_run` 和
`previous_elapsed_seconds`（从此前关联 W&B 运行读取实际累计耗时，不能清零）。
继续使用原分词器、数据、模型和优化器配置；程序不再自动核对其哈希和配置一致性。
每次启动仍重新编码原文，耗时计入累计预算，不使用免计时的开发缓存。

测试：`uv run pytest`。提交一份 Markdown 文档，包含姓名、学号、两项正式 W&B 链接，
有续训时附关联运行。此前 PPL 64.75 是验证前缀的开发成绩，不能当作完整验证成绩。

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
