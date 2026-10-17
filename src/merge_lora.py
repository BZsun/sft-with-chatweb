#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/merge_lora.py
=================
把训练好的 LoRA 适配器合并到基座模型，导出为单一 HF 格式模型，
便于部署和推理（不需要基座模型路径）。

用法:
    python src/merge_lora.py
    python src/merge_lora.py --adapter ./outputs/lora_adapter --out ./outputs/merged_model
    python src/merge_lora.py --quantization q4_k_m    # GGUF 量化（需 llama.cpp）
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from unsloth import FastLanguageModel

DEFAULT_BASE_MODEL = "/ethan/pretrained_model/Qwen3-8B"
DEFAULT_ADAPTER = "./outputs/lora_adapter"
DEFAULT_OUT = "./outputs/merged_model"
MAX_SEQ_LENGTH = 2048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并 LoRA 到 HF 模型")
    p.add_argument("--base-model", type=str, default=os.getenv("BASE_MODEL_PATH", DEFAULT_BASE_MODEL))
    p.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER, help="LoRA 适配器目录")
    p.add_argument("--out", type=str, default=os.getenv("MERGED_MODEL_PATH", DEFAULT_OUT),
                   help="合并后模型输出目录（默认取 .env 的 MERGED_MODEL_PATH）")
    p.add_argument("--quantization", type=str, default=None,
                   choices=[None, "q4_k_m", "q8_0", "f16"],
                   help="可选量化（仅在 save_method=gguf 时生效）")
    p.add_argument("--save-method", type=str, default="merged_16bit",
                   choices=["merged_16bit", "merged_4bit", "lora"],
                   help="保存方式: merged_16bit(fp16,~15GB), merged_4bit(~5GB), lora(只 adapter)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    out_path = Path(args.out)
    if not adapter_path.exists():
        raise FileNotFoundError(f"找不到 LoRA 适配器: {adapter_path}")
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 加载基座 + 适配器 ...")
    print(f"  base: {args.base_model}")
    print(f"  adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),  # Unsloth 会自动加载 base + adapter
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"[2/2] 导出到 {out_path} (method={args.save_method}) ...")
    if args.save_method == "lora":
        # 只保存 adapter
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
    else:
        # merged 模式
        model.save_pretrained_merged(
            str(out_path),
            tokenizer,
            save_method=args.save_method,  # "merged_16bit" / "merged_4bit"
        )

    print(f"\n✓ 已保存到 {out_path}")
    # 显示产出大小
    total = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
    print(f"  体积: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
