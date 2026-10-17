#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py — 项目统一入口（替代原 scripts/*.sh）

用法:
    python run.py data                 # 构建 SFT 数据集
    python run.py train                # 正式训练
    python run.py train --dry-run      # Dry-run 验证流水线（10 步）
    python run.py merge                # 合并 LoRA → HF 模型
    python run.py serve                # 启动推理服务（默认 0.0.0.0:8118）

说明:
    - 所有子命令通过子进程调用 src/ 下脚本（unsloth 需独立进程先于 transformers 导入，
      训练显存也只在子进程退出后释放）
    - 自动加载项目根目录 .env（已 export 的环境变量优先）
    - data/train/merge 的额外参数原样透传，见各自 --help（如 python src/train.py --help）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def _py() -> str:
    return sys.executable


# ---- 子命令 ----

def cmd_data(rest: list[str]) -> int:
    return subprocess.call([_py(), "src/build_dataset.py", *rest], cwd=ROOT)


def cmd_train(rest: list[str]) -> int:
    dry_run = "--dry-run" in rest
    if dry_run and not (ROOT / "data/processed/_dryrun.jsonl").exists():
        print("[run.py] 未找到 data/processed/_dryrun.jsonl，先生成 dry-run 数据 ...")
        code = subprocess.call([_py(), "src/build_dataset.py", "--dryrun", "200"], cwd=ROOT)
        if code != 0:
            return code

    cmd = [_py(), "-u", "src/train.py", *rest]
    if dry_run or "--help" in rest or "-h" in rest:
        return subprocess.call(cmd, cwd=ROOT)

    log_file = ROOT / "outputs/logs" / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run.py] 日志同步写入: {log_file}")

    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    with log_file.open("w", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
    return proc.wait()


def cmd_merge(rest: list[str]) -> int:
    return subprocess.call([_py(), "src/merge_lora.py", *rest], cwd=ROOT)


def cmd_serve() -> int:
    """一键启动：OpenAI API + Web 聊天界面（src/modelServer.py 同端口托管）。"""
    merged = Path(os.getenv("MERGED_MODEL_PATH", "./finetune/Qwen3-8B-finetuned"))
    lora = Path(os.getenv("LORA_ADAPTER_PATH", "./outputs/lora_adapter"))
    base = Path(os.getenv("BASE_MODEL_PATH", "/ethan/pretrained_model/Qwen3-8B"))

    has_merged = (merged / "config.json").is_file()
    has_lora = (lora / "adapter_config.json").is_file() and (base / "config.json").is_file()
    has_base = (base / "config.json").is_file()

    if not (has_merged or has_lora or has_base):
        print("[ERROR] 找不到可用模型:", file=sys.stderr)
        print(f"        MERGED_MODEL_PATH={merged}", file=sys.stderr)
        print(f"        LORA_ADAPTER_PATH={lora} + BASE_MODEL_PATH={base}", file=sys.stderr)
        print("        请先完成训练并执行 python run.py merge，或在 .env 配置正确路径", file=sys.stderr)
        return 1

    print("=" * 42)
    print("  一键启动：OpenAI API + Web 聊天界面")
    print("=" * 42)
    if has_merged:
        print(f"  模型: {merged} [合并微调模型]")
    elif has_lora:
        print(f"  模型: {lora} [基座+LoRA]")
    else:
        print(f"  模型: {base} [基座]")
    print()

    return subprocess.call([_py(), "src/modelServer.py"], cwd=ROOT)


# ---- 主入口 ----

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Medical-SFT 统一入口（数据 → 训练 → 合并 → 服务）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
子命令:
  python run.py data     构建 SFT 数据集（参数透传给 src/build_dataset.py）
  python run.py train    SFT 训练（--dry-run / --resume-from P / --epochs N / --max-seq-length N）
  python run.py merge    合并 LoRA 到 HF 模型（--save-method merged_16bit|merged_4bit|lora）
  python run.py serve    一键启动 OpenAI API + Web 聊天界面（默认 0.0.0.0:8118）

示例:
  python run.py data --n-per-dept 10000
  python run.py train --dry-run
  python run.py serve
""")
    parser.add_argument("command", choices=["data", "train", "merge", "serve"], help="子命令")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    _load_env()

    if args.command == "serve":
        return cmd_serve()
    return cmd_data(args.rest) if args.command == "data" \
        else cmd_train(args.rest) if args.command == "train" \
        else cmd_merge(args.rest)


if __name__ == "__main__":
    sys.exit(main())
