import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

# 实验二入口：计时从重型依赖导入之前开始，涵盖编码、训练、保存和验证。
# ruff: noqa: E402
from contextlib import nullcontext
from datetime import UTC, datetime
from itertools import islice
import json
import math
from pathlib import Path

import numpy as np
import torch

from bpe import BPE
from transformer_module import AdamW, TransformerLM, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping
from wandb_record import experiment

