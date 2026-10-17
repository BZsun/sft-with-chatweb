#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/build_dataset.py
====================
将 6 个科室的 GBK 编码 CSV 转换为统一 UTF-8 JSONL (Qwen ChatML messages 格式)，
供 src/train.py 使用。

流程:
    1. 扫描 6 个科室目录，定位每个目录下的 *5-*.csv / *6-*.csv / *.csv
    2. 用编码回退链 utf-8-sig → gbk → gb2312 → gb18030 读 CSV
    3. 字段归一化: ask(question/问题), answer(回答/response)
    4. 清洗: 去空、去 (ask+answer) 超过 MAX_CHAR_LEN 的行
    5. 去重: 按 hash(ask+answer) 去重
    6. 均衡采样: 每科 N_PER_DEPT 条
    7. Shuffle + 98/2 切 train/val
    8. 输出 JSONL: {"messages":[system,user,assistant]}，UTF-8

用法:
    python src/build_dataset.py
    python src/build_dataset.py --raw ./datasets --out ./data/processed --n-per-dept 20000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# ---- 常量 ----

DEPARTMENTS = {
    "IM_内科": "内科",
    "Surgical_外科": "外科",
    "Pediatric_儿科": "儿科",
    "Oncology_肿瘤科": "肿瘤科",
    "OAGD_妇产科": "妇产科",
    "Andriatria_男科": "男科",
}

# ask 字段候选（按优先级匹配）
ASK_KEYS = ("ask", "question", "问题", "ask_text", "query")
# answer 字段候选
ANSWER_KEYS = ("answer", "response", "回答", "reply", "answer_text")

SYSTEM_PROMPT = (
    "你是一名专业的中文医疗助手，请基于医学知识为患者提供准确、严谨的回答，"
    "并在必要时建议就医。"
)

# 编码回退顺序（utf-8-sig 放最前，兼容 Excel 导出）
ENCODINGS = ("utf-8-sig", "gbk", "gb2312", "gb18030")


# ---- 编码感知读取 ----

def read_csv_with_encoding(path: Path) -> list[dict]:
    """按 ENCODINGS 顺序尝试读取 CSV，返回 list of dict。
    抛出 RuntimeError 当所有编码都失败时。"""
    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                return rows
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
        except Exception as e:
            # 解析错误不算编码问题
            last_err = e
            continue
    raise RuntimeError(f"无法用任何编码读取 {path}，最后错误: {last_err}")


def get_field(row: dict, keys: tuple[str, ...]) -> str:
    """从行 dict 中按候选键名取首个非空字段值。"""
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return ""


# ---- 单文件处理 ----

def iter_rows_for_dept(dept_dir: Path) -> Iterator[dict]:
    """对单个科室目录，遍历其下所有 CSV，yield 归一化后的行。
    归一化字段: department, ask, answer
    """
    if not dept_dir.exists():
        print(f"  [WARN] 目录不存在: {dept_dir}", file=sys.stderr)
        return
    csv_files = sorted(p for p in dept_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_files:
        print(f"  [WARN] 未找到 CSV: {dept_dir}", file=sys.stderr)
        return

    for csv_path in csv_files:
        print(f"  读取 {csv_path.name} ...", end="")
        try:
            rows = read_csv_with_encoding(csv_path)
        except RuntimeError as e:
            print(f" 失败: {e}", file=sys.stderr)
            continue
        kept = 0
        for row in rows:
            ask = get_field(row, ASK_KEYS)
            answer = get_field(row, ANSWER_KEYS)
            if not ask or not answer:
                continue
            yield {
                "ask": ask,
                "answer": answer,
            }
            kept += 1
        print(f" {kept} 行")


# ---- 主流程 ----

def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    n_per_dept: int,
    max_char_len: int,
    seed: int,
) -> dict:
    """执行完整数据构建流程，返回统计信息 dict。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    stats: dict = {
        "n_per_dept": n_per_dept,
        "max_char_len": max_char_len,
        "seed": seed,
        "departments": {},
        "total_after_dedup": 0,
        "total_after_sample": 0,
        "train_size": 0,
        "val_size": 0,
    }

    # Step 1: 读 + 清洗 + 去重 + 长度过滤
    dept_samples: dict[str, list[dict]] = {}
    for dept_key, dept_name in DEPARTMENTS.items():
        dept_dir = raw_dir / dept_key
        print(f"\n[{dept_name}] {dept_dir}")
        rows = list(iter_rows_for_dept(dept_dir))
        raw_count = len(rows)

        # 长度过滤
        rows = [r for r in rows if len(r["ask"]) + len(r["answer"]) <= max_char_len]
        after_len = len(rows)

        # 去重
        seen: set[str] = set()
        uniq: list[dict] = []
        for r in rows:
            h = hashlib.md5((r["ask"] + r["answer"]).encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            uniq.append(r)
        after_dedup = len(uniq)

        # 均衡采样（不足则全取）
        if len(uniq) >= n_per_dept:
            sampled = rng.sample(uniq, n_per_dept)
        else:
            print(f"  [WARN] {dept_name} 仅 {len(uniq)} 条，少于 {n_per_dept}，全取", file=sys.stderr)
            sampled = uniq

        dept_samples[dept_name] = sampled
        stats["departments"][dept_name] = {
            "raw": raw_count,
            "after_length": after_len,
            "after_dedup": after_dedup,
            "sampled": len(sampled),
            "avg_ask_chars": round(sum(len(r["ask"]) for r in sampled) / max(len(sampled), 1), 1),
            "avg_answer_chars": round(sum(len(r["answer"]) for r in sampled) / max(len(sampled), 1), 1),
        }
        stats["total_after_dedup"] += after_dedup
        stats["total_after_sample"] += len(sampled)
        print(
            f"  raw={raw_count}, after_len={after_len}, after_dedup={after_dedup}, sampled={len(sampled)}"
        )

    # Step 2: 合并 + shuffle
    all_samples: list[dict] = []
    for samples in dept_samples.values():
        all_samples.extend(samples)
    rng.shuffle(all_samples)

    # Step 3: 切 train/val
    val_size = max(1, int(len(all_samples) * 0.02))
    val_set = all_samples[:val_size]
    train_set = all_samples[val_size:]
    stats["train_size"] = len(train_set)
    stats["val_size"] = len(val_set)

    # Step 4: 序列化为 JSONL
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_set:
            obj = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["ask"]},
                    {"role": "assistant", "content": r["answer"]},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val_set:
            obj = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["ask"]},
                    {"role": "assistant", "content": r["answer"]},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Step 5: 写 stats.json
    stats_path = out_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] train={len(train_set)} → {train_path}")
    print(f"[DONE] val  ={len(val_set)} → {val_path}")
    print(f"[DONE] stats → {stats_path}")
    return stats


# ---- Dry-run 数据 ----

def build_dryrun_dataset(out_dir: Path, n: int = 200) -> Path:
    """从 train.jsonl 头部取 n 条作为 dryrun 集，方便快速验证训练流水线。"""
    train_path = out_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"请先运行完整 build_dataset 流程生成 {train_path}")
    dryrun_path = out_dir / "_dryrun.jsonl"
    with train_path.open("r", encoding="utf-8") as fin, dryrun_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
    print(f"[DRYRUN] 写 {n} 条 → {dryrun_path}")
    return dryrun_path


# ---- CLI ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="中文医疗数据集预处理")
    p.add_argument("--raw", type=Path, default=Path(os.getenv("RAW_DATA_DIR", "./datasets")),
                   help="原始数据根目录（含 6 个科室子目录）")
    p.add_argument("--out", type=Path, default=Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed")),
                   help="处理后输出目录")
    p.add_argument("--n-per-dept", type=int, default=int(os.getenv("N_PER_DEPT", "20000")), help="每科采样条数")
    p.add_argument("--max-char-len", type=int, default=int(os.getenv("MAX_CHAR_LEN", "600")), help="ask+answer 总字符上限")
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")), help="随机种子")
    p.add_argument("--dryrun", type=int, default=0,
                   help="从已有 train.jsonl 抽取 N 条作为 dry-run 集（需先跑过完整流程）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dryrun and args.dryrun > 0:
        build_dryrun_dataset(args.out, n=args.dryrun)
        return
    build_dataset(
        raw_dir=args.raw,
        out_dir=args.out,
        n_per_dept=args.n_per_dept,
        max_char_len=args.max_char_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
