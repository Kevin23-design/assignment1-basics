# BPE 与 Transformer 作业

只保留正式实验所需的实现，单入口启动，不需要开发数据准备或实验编排脚本。

| 文件 | 用途 |
|---|---|
| `bpe.py` | 自己的字节级 BPE：训练、编码、解码、流式接口和保存加载 |
| `train_bpe.py` | 任务一训练与完整验证测速、前 200 个 token 展示 |
| `transformer_module.py` | 手写 Transformer、交叉熵、AdamW、学习率、梯度裁剪与采样 |
| `transformer_main.py` | 任务二唯一入口：完整编码、训练、保存恢复、完整验证和计时 |
| `wandb_record.py` | 保持课程提供的源码记录器不变 |

官方测试、适配器和环境文件保留。历史 `outputs/`、`wandb/`、已训练分词器和数据不删除；
这些是实验产物，不是当前运行依赖的开发分支。源码快照保留供历史实验复查。

## 环境与 W&B

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

网络不可用时设 `WANDB_MODE=offline`，之后用 `uv run wandb sync <离线run目录>` 同步。
提交前确认助教有权限查看两个正式运行。

## 运行

任务一：默认完整读取 `data/owt_train.txt`、`data/owt_valid.txt`，词表上限 32,000，
特殊标记为 `<|endoftext|>`。可用 `--train-file`、`--valid-file` 指定课程公共路径。

```sh
uv run python train_bpe.py
```

任务二：在 A100/A800 80GB 上，从零初始化训练，唯一启动入口为：

```sh
uv run python transformer_main.py
```

默认使用现有本人分词器 `outputs/830f59c92e15/tokenizer`。迁移服务器时，需要带上
该前缀对应的 `.vocab.json`、`.merges.json`、`.config.json`，并准备完整原始 OWT 文件。
这是单入口，不是无需依赖的独立 Python 文件。公共路径不同可显式指定：

```sh
uv run python transformer_main.py \
  --train-file /data/nlp_course/1-basic-data/owt_train.txt \
  --valid-file /data/nlp_course/1-basic-data/owt_valid.txt \
  --bpe-prefix outputs/830f59c92e15/tokenizer
```

默认模型：4 层、512 维、8 头、FFN 1408、上下文 512、BF16。
微批次 64、梯度累积 1，即每次更新 32,768 token；这是面向 80GB 卡的候选配置，尚未实测。
本地 4080 SUPER 若要保持相同有效 batch，可覆盖为 `--batch-size 16 --gradient-accumulation 4`。
AdamW β=(0.9,0.999)、ε=1e-8、权重衰减 0.1、梯度裁剪 1.0。
学习率默认 3e-4→3e-5，warmup 200 步，余弦终点 20,000 步，种子 42。
调度参数是待校准起点，不是已验证的最优配置；在 A100 上测得吞吐后再确定余弦终点。
默认不限制总步数，按剩余时间收尾；超过余弦终点后保持最低学习率。

学习率对照只需改 `--learning-rate 1e-3`，其余条件应相同；可用 `--max-steps` 限制短程更新数。
没有自动启动任何对照实验。所有运行仍读取指定文件全文，没有字符截断或开发清单接口。

## 与作业功能要求核对

| 要求 | 当前行为 |
|---|---|
| 本人 32k BPE | 加载自己的词表、merges、特殊标记；模型使用实际词表大小 |
| BPE 正确性与测速 | UTF-8、保留空白；新实例全文 encode/decode/比较；展示在计时外 |
| BPE 保存和流式接口 | 保留；全文编码缓存固定 1,048,576 项，满后清空；流式内部容量维持原实现 |
| 手写组件 | Linear/Embedding/RMSNorm/SwiGLU/RoPE/注意力/交叉熵/AdamW/调度/裁剪均为原手写实现 |
| 完整 OWT 训练 | 启动后用本人 BPE 编码原文，保存磁盘数组，从整个数组随机抽连续窗口 |
| 不额外插入特殊标记 | 保留原始文本及原标记，不插入 BOS/EOS/PAD |
| 六小时累计预算 | 最顶部开始计时，包含导入、W&B、源码、编码、初始化、训练、保存和最终验证 |
| 为完整验证留时间 | 测量少量验证批次，按全文规模估计并留安全余量；超时明确报错 |
| 最后模型评分 | 仅保存本地 `last.pt`，随后对该模型验证，不选择历史最佳检查点 |
| 完整验证 PPL | eval/no_grad；总目标 NLL 除以目标总数再取 exp；覆盖尾部，仅首 token 无预测目标 |
| 低频训练日志 | 每 100 次更新记录区间 PPL，末尾不足 100 步也记录；设备上累计日志损失 |
| W&B 源码与评分 | 自动打印、快照并上传源码；记录最终 `val/ppl`、有效目标数、步数、UTC 时间及累计耗时 |
| 保存恢复 | 模型、优化器、步数、随机状态、数据和分词器指纹；续训不得重置预算 |

原来独立开发准备的 27.27 亿 token 缓存不直接作为免计时的正式输入。
正式路径重新编码完整原文，数据规模以此次准确编码数量为准。完整编码耗时占用六小时预算。
验证必须用完整 `owt_valid.txt`；此前 64.75 是验证前缀的开发成绩，不能当作正式成绩。
W&B 最后汇总同步和进程退出不计入课程计算预算；模型权重只保存在本地。

续训时沿用原配置，并追加以下参数：

```sh
--resume outputs/<原run_id>/last.pt \
--resume-from-run <原run_id> \
--previous-elapsed-seconds <原timing.json记录的累计秒数>
```

会核对来源、配置、数据与实际累计耗时。旧开发分支检查点不用于这条正式路径。

## 测试与提交

```sh
uv run pytest
```

官方测试通过 `tests/adapters.py` 对接，未修改测试预期；另保留正式入口的少量回归测试。
最终提交一份 Markdown 文档，填写姓名、学号及 BPE、Transformer 的正式 W&B 链接；
如有续训，附关联运行。正式六小时实验与 A100 吞吐校准尚未执行，不能把开发成绩冒充正式成绩。
