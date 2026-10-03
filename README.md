# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

BPE tests are connected through [tests/adapters.py](./tests/adapters.py).
The remaining Transformer adapters are still to be implemented, so running the
entire suite is not expected to pass yet. Use the focused BPE command below.

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


## Course byte-level BPE

`bpe.py` implements the course `BPE` interface. Run the focused tests with:

```sh
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py
```

Train and evaluate on the local OWT files (this starts a W&B run):

```sh
export WANDB_ENTITY="your-account-or-team"
export STUDENT_ID="your-student-id"
export WANDB_PROJECT="lmfs-assignment1"
uv run python train_bpe.py
```

The defaults are `data/owt_train.txt`, `data/owt_valid.txt`, a 32,000-entry
vocabulary, and `--special-tokens '<|endoftext|>'`. Special tokens are recognized
literally; they are not automatically inserted. Pass `--special-tokens` with no
values for an empty list. A small local smoke run without uploading anything is:

```sh
WANDB_MODE=offline uv run python train_bpe.py --train-file tests/fixtures/corpus.en --valid-file tests/fixtures/corpus.en --vocab-size 500
```

The run writes `outputs/<run-id>/tokenizer.{vocab,merges,config}.json`.
Vocabulary bytes and merge components use hexadecimal strings. `load(prefix)`
restores the configuration and special IDs; `from_files(vocab_path, merges_path,
special_tokens)` reads these hex JSON files with an explicitly supplied special
list. It does not read GPT-2's byte-to-Unicode JSON/text format.

Evaluation loads a fresh tokenizer, reads text with `newline=""`, then times
only encoding, decoding, and equality comparison. Tables and source recording
are outside that interval. Use the course-provided `wandb_record.py` unchanged.
`WANDB_ENTITY` and `STUDENT_ID` are required even for offline runs;
`WANDB_PROJECT` defaults to `lmfs-assignment1`. Set `WANDB_MODE=offline`
to record locally for later synchronization with `uv run wandb sync <run-dir>`.
Do not submit debugging runs as formal OWT results.

The recorder prints numbered source lines with SHA256 hashes and saves the
source snapshot and `source_dump.txt` in `outputs/<run-id>/`. It uploads a
`source-code` artifact and checks for source changes when the experiment exits.
The training entry point explicitly passes the project root, so source discovery
and default data/output paths do not depend on the current working directory.
Custom data paths are interpreted relative to the current working directory.

Training reads bounded text chunks and maintains counts over distinct
pretokens, plus an inverted pair index. Its memory use still scales with the
number of distinct pretokens and pairs. Streaming encoding retains unresolved
suffixes across chunks, including special-token prefixes. An arbitrarily long
unfinished pretoken can require a correspondingly long buffer. Encoding caches
are bounded and reset when training or loading a tokenizer.

### Fixed validation cache

Full-text encoding uses only a 1,048,576-entry dictionary cache, cleared when
full. There are no cache-policy or capacity options. Run the formal entry with:

```sh
uv run python train_bpe.py
```

Loading a tokenizer starts with an empty cache and needs no extra configuration.
The entry point records the fixed policy and capacity in W&B config. Streaming
encoding uses the same clear-all logic with a private 2,048-entry working-memory
bound; it is not an alternative full-text configuration.

## 实验二：手写 Transformer

核心实现位于 `transformer_module.py`，训练入口为 `transformer_main.py`。
包括无偏置 Linear、Embedding、RMSNorm、SiLU/SwiGLU、RoPE、因果多头注意力、
Pre-norm Transformer、稳定交叉熵、AdamW、余弦学习率和全局梯度裁剪。
官方测试通过 `tests/adapters.py` 调用这些实现，测试预期未修改。

```sh
uv run pytest
```

沿用任务一已经训练的分词器；`--bpe-prefix` 是实际保存路径前缀，不包含文件后缀。
下面是当前项目产物对应的运行示例，正式训练前可调整模型与批次配置：

```sh
uv run python transformer_main.py \
  --bpe-prefix outputs/348a7e489d53/tokenizer \
  --device cuda --precision bf16
```

W&B 环境变量与任务一相同。默认读取完整 `data/owt_train.txt`、`data/owt_valid.txt`，
使用分词器实际词表大小，保持其特殊 token 配置，不额外插入 BOS/EOS/PAD。
所有模型权重随机初始化。默认模型为 8 层、512 维、8 个头、1408 维前馈层，
上下文 256；训练微批次 8、累积 4 次梯度。默认超参数是起点，尚未经过正式六小时训练调优。

无需上传的小规模 CUDA 验证（使用同一份正式分词器，文本是测试样本）：

```sh
WANDB_MODE=disabled WANDB_ENTITY=local STUDENT_ID=smoke \
uv run python transformer_main.py \
  --bpe-prefix outputs/348a7e489d53/tokenizer \
  --train-file tests/fixtures/corpus.en --valid-file tests/fixtures/corpus.en \
  --device cuda --precision bf16 \
  --d-model 32 --num-layers 1 --num-heads 4 --d-ff 64 \
  --context-length 32 --batch-size 2 --eval-batch-size 4 \
  --gradient-accumulation 2 --warmup-steps 0 --max-steps 5
```

`--max-steps` 是累计更新次数上限；省略时按剩余预算停止训练。
每 100 次参数更新记录区间训练 PPL，末尾不足 100 步的区间也记录。
计时从入口导入 PyTorch 之前开始，包含源码记录、数据编码、初始化、训练、保存和全量验证。
默认至少预留 1200 秒；程序还会实测少量验证批次，根据全量验证耗时估计扩大预留。
预留是估计，超时会报错并标记 `time/within_6h=false`，不会把不完整验证当作评分结果。

训练与验证文本通过本人 BPE 的流式接口编码，保持跨读取块的分词边界，
编码结果保存为本地磁盘数组，避免一次性把完整 OWT token 列表放入内存。
每次运行（含续训）重新编码，耗时均计入预算。
验证按不重叠的上下文窗口覆盖全部目标位置；仅全文件首 token 没有前文，不计为目标。
窗口内使用因果注意力，尾部不足一个窗口的 token 仍计分；组批补位不计损失。
最终 PPL 为 `exp(全验证集 NLL 总和 / 有效目标 token 数)`。

本地 `outputs/<run-id>/` 保存 `last.pt`、`timing.json`、`result.json`、
数据哈希及 token 数组，以及记录器生成的源码快照。检查点包含模型、优化器、
更新次数和随机数状态；只保存最后模型，随后对该模型验证，不上传模型文件。

续训需保留原运行的 `last.pt` 和同目录 `timing.json`，沿用原模型、优化器、
学习率调度和采样配置。例如在原命令末尾追加：

```sh
--resume outputs/<原run-id>/last.pt \
--resume-from-run <原run-id> \
--previous-elapsed-seconds <原timing.json中的time/total_seconds>
```

续训检查 BPE、原始数据哈希和配置一致性，恢复随机数状态；累计预算不会重置。
调试结果仅用于验证代码，不作为正式 OWT 评分结果。
