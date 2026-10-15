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
普通预分词片段。正式入口保持课程提供的固定配置与 encode/decode 调用流程。
流式 `encode_iterable()` 继续使用独立的 2,048 项缓存。

云端三轮缓存探索中，1M 清空策略中位数为 41.499 秒；1M 停止新增为 41.728 秒，
1M LRU 为 51.577 秒，全部通过完整 ID 对照与全文还原。1M 实际缓存 627,462 项，
未触发容量上限，因此本次无法比较它与停止新增策略在满后的行为。当前继续保留原策略。
以上是探索结果，最终提交仍使用正式 W&B 验证记录。

任务一正式入口 `train_bpe.py` 按课程模板保留：先训练并保存、重新加载、读取全文，
再计时执行 `bpe.encode(text)`、`bpe.decode(ids)` 和原文比较；前 200 token 展示及上传
在计时后。数据路径固定为公共 OWT 原文，运行 `python train_bpe.py`，不附加路径或 worker
参数。最优参数现已封装在 `BPE` 内部：**16 workers / 256 个安全分片 / 131,072 IDs 解码块**。
`encode()` 仍返回完整 `list[int]`，`decode()` 先拼接全部 bytes，再进行一次 UTF-8 解码，
非法序列使用 `errors="replace"`。不修改公开接口或特殊 token 的序列化格式。

大于等于 8 Mi 字符的完整文本在 Linux 上自动尝试并行；切点只能在分词器实际识别到的
完整特殊 token 后，遵守较长标记优先，不限于某个写死的标记。OWT 默认特殊清单仅包含
`<|endoftext|>`，其 256 分片口径与探索相同。小输入、没有足够安全切点、非 Linux 环境
或超出 uint32 范围的自定义 ID 保留串行编码行为。能够安全分片而允许 CPU 不足 16 个时
明确报错，不自动减核；worker 运行失败也不会悄悄串行重跑。

父进程及记录服务限定在前 16 个允许的逻辑 CPU 内，worker 各固定一个。独立轻量协调
入口使用 `multiprocessing` 的 spawn 上下文启动 worker，避免重复导入模板顶层的 W&B；
子进程启动环境在导入前设置底层线程为 1、隐藏 GPU，调用者的 GPU 可见性和线程环境
变量不变。worker 从当前词表、合并规则和完整特殊清单重建空缓存 BPE，不依赖磁盘前缀
继续存在，不缓存进程池。原文、分片和 uint32 结果均通过内存 IPC，不生成临时 token 文件；
协调、启动、状态传输、全部 worker 退出、有序列表组装都包含在 `encode()` 中。

配对探索中位数从 12.608628 秒降到 12.012227 秒（耗时降低 4.73%，5/5 对胜出）。
**12.012227 秒仍是辅助实现的探索结果，不代表原样正式入口的新成绩。** 兼容模板新增的
轻量协调与传输也会计入正式耗时，必须在云端用 `python train_bpe.py` 重新完成训练、
加载、编码、解码和比较，再以本次 W&B 的 `bpe/valid_seconds` 评分。
探索工具继续保留在 `exploration` 分支，任务二的流式编码与训练配置不变。

本地仅小文本检查：

```sh
python -m pytest tests/test_bpe_memory_runtime.py tests/test_tokenizer.py tests/test_train_bpe.py -q
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
