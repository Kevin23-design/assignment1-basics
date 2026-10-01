import argparse
import json
import time
from pathlib import Path
import wandb
from wandb_record import experiment
from bpe import CACHE_CAPACITY


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate byte-level BPE")
    parser.add_argument("--train-file", default=str(root / "data/owt_train.txt"))
    parser.add_argument("--valid-file", default=str(root / "data/owt_valid.txt"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    args = parser.parse_args()
    config = {
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "vocab_size": args.vocab_size,
        "special_tokens": args.special_tokens,
        "cache_policy": "clear_all",
        "cache_eviction": "clear_all",
        "cache_capacity": CACHE_CAPACITY,
        "cache_max_piece_characters": 128,
    }
    with experiment("bpe", config, root=root) as run:
        from bpe import BPE

        bpe = BPE()
        bpe.train(config["train_file"], config["vocab_size"], config["special_tokens"])
        prefix = str(root / "outputs" / run.id / "tokenizer")
        bpe.save(prefix)
        # 重新加载，避免使用训练或先前评估产生的编码缓存。
        bpe = BPE()
        bpe.load(prefix)
        print(f"验证缓存：满后清空，容量：{CACHE_CAPACITY}", flush=True)
        with open(config["valid_file"], encoding="utf-8", newline="") as f:
            text = f.read()
        started = time.perf_counter()
        ids = bpe.encode(text)
        decoded_text = bpe.decode(ids)
        passed = decoded_text == text
        valid_seconds = time.perf_counter() - started
        # 以下均为展示/审查操作，不计入验证耗时。
        raw = text.encode("utf-8")
        first_ids = ids[:200]
        rows, offset = [], 0
        for i, token_id in enumerate(first_ids):
            piece = bpe.vocab[token_id]
            if not isinstance(piece, bytes):
                raise TypeError("vocab 的值必须是 bytes")
            end = offset + len(piece)
            try:
                piece.decode("utf-8", errors="strict")
                standalone_valid = True
            except UnicodeDecodeError:
                standalone_valid = False
            rows.append(
                [
                    i,
                    int(token_id),
                    piece.hex(),
                    repr(piece),
                    repr(bpe.decode([token_id])),
                    standalone_valid,
                    offset,
                    end,
                    piece == raw[offset:end],
                ]
            )
            offset = end
        columns = [
            "position",
            "token_id",
            "token_hex",
            "token_bytes",
            "decoded_token",
            "standalone_utf8_valid",
            "byte_start",
            "byte_end",
            "matches_input_bytes",
        ]
        print("前 200 个 encode ID：", first_ids, flush=True)
        print("前 200 个 token 整体 decode：", repr(bpe.decode(first_ids)), flush=True)
        print("逐 token 详情字段：", columns, flush=True)
        for row in rows:
            print(row, flush=True)
        run.log({"bpe/first_200_tokens": wandb.Table(columns=columns, data=rows)})
        result = {
            "bpe/valid_seconds": valid_seconds,
            "bpe/valid_token_count": len(ids),
            "bpe/roundtrip_passed": passed,
            "bpe/vocab_size": len(bpe.vocab),
        }
        run.summary.update(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print("任务二加载本人 BPE 的路径前缀：", prefix, flush=True)
        if len(ids) < 200 or not passed:
            raise ValueError("评估文本不足 200 个 token 或还原失败，请检查输入与实现")


if __name__ == "__main__":
    main()
