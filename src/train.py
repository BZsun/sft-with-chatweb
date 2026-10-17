#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/train.py
============
Unsloth + LoRA (QLoRA 4-bit) 微调 Qwen3-8B。
使用 Qwen ChatML 模板 + assistant_only_loss，仅在 assistant 段计算 loss。

相对素材原脚本的改进:
    1. 路径走环境变量 BASE_MODEL_PATH，可移植
    2. 文本格式化走 tokenizer.apply_chat_template，贴近 Qwen Instruct 预训练格式
    3. SFTConfig 启用 assistant_only_loss=True（trl>=0.12）
    4. packing=True，3× 加速
    5. checkpoint: save_steps=500, save_total_limit=2
    6. eval: eval_strategy=steps, eval_steps=500（val 2%）
    7. --dry-run 模式：取 _dryrun.jsonl 前 200 条跑 max_steps=10
    8. --resume-from <path> 断点续训

用法:
    # Dry-run（先验证流水线）
    python src/train.py --dry-run

    # 正式训练
    python src/train.py

    # 断点续训
    python src/train.py --resume-from outputs/lora_adapter/checkpoint-500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Unsloth 必须先于 transformers / trl / peft 导入，否则优化不生效
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ---- 常量 ----

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LEARNING_RATE = 2e-4
# Qwen3-8B 4bit + 3090 24GB 实测峰值约 18GB；若 OOM：本值降为 1，并把 GRADIENT_ACCUMULATION_STEPS 翻倍到 8
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # effective 8
NUM_TRAIN_EPOCHS = 1
WARMUP_STEPS = 20
SAVE_STEPS = 200
EVAL_STEPS = 200
LOGGING_STEPS = 10
SEED = 3407
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.01
PACKING = True
DATASET_NUM_PROC = 2

DEFAULT_BASE_MODEL = "/ethan/pretrained_model/Qwen3-8B"
DEFAULT_DATA_DIR = "./data/processed"
DEFAULT_OUTPUT_DIR = "./outputs/lora_adapter"
DEFAULT_LOGS_DIR = "./outputs/logs"


# ---- CLI ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-8B 医疗 LoRA 微调")
    p.add_argument("--base-model", type=str, default=os.getenv("BASE_MODEL_PATH", DEFAULT_BASE_MODEL),
                   help="基座模型路径或 HF 仓库名")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                   help="含 train.jsonl / val.jsonl 的目录")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="LoRA 权重输出目录")
    p.add_argument("--logs-dir", type=str, default=DEFAULT_LOGS_DIR,
                   help="训练日志目录")
    p.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS, help="训练 epoch 数")
    p.add_argument("--max-steps", type=int, default=-1, help="最大训练步数，-1 表示由 epoch 决定")
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="最大序列长度")
    p.add_argument("--dry-run", action="store_true",
                   help="用 _dryrun.jsonl 跑少量步数验证流水线")
    p.add_argument("--resume-from", type=str, default=None,
                   help="从指定 checkpoint 续训（传入 checkpoint 路径）")
    return p.parse_args()


# ---- 训练主流程 ----

def main() -> None:
    args = parse_args()

    # Dry-run 模式：把数据文件换成 _dryrun.jsonl
    if args.dry_run:
        train_file = Path(args.data_dir) / "_dryrun.jsonl"
        val_file = Path(args.data_dir) / "_dryrun.jsonl"
        if not train_file.exists():
            print(f"[ERROR] Dry-run 数据不存在: {train_file}\n"
                  f"        请先执行: python src/build_dataset.py --dryrun 200",
                  file=sys.stderr)
            sys.exit(1)
        max_steps = 10
        epochs = 1
        save_steps = 999999  # dry-run 不存 checkpoint
        eval_steps = 999999
    else:
        train_file = Path(args.data_dir) / "train.jsonl"
        val_file = Path(args.data_dir) / "val.jsonl"
        if not train_file.exists():
            print(f"[ERROR] 找不到训练数据: {train_file}\n"
                  f"        请先执行: python run.py data",
                  file=sys.stderr)
            sys.exit(1)
        max_steps = args.max_steps
        epochs = args.epochs
        save_steps = SAVE_STEPS
        eval_steps = EVAL_STEPS

    # 路径准备
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    log_file = Path(args.logs_dir) / ("train_dryrun.log" if args.dry_run else "train.log")

    print(f"[INFO] 基座模型: {args.base_model}")
    print(f"[INFO] 训练数据: {train_file}")
    print(f"[INFO] 验证数据: {val_file}")
    print(f"[INFO] 输出目录: {args.output_dir}")
    print(f"[INFO] 日志文件: {log_file}")
    print(f"[INFO] 模式: {'DRY-RUN' if args.dry_run else 'NORMAL'}")

    # Step 1: 加载基座模型 (Unsloth + 4-bit)
    print("\n[1/4] 加载基座模型 ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # 自动选 bf16 / fp16
        load_in_4bit=True,
    )

    # Step 2: 注入 LoRA
    print("[2/4] 注入 LoRA 适配器 ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% 显存节省
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    # Step 3: 加载并格式化数据
    print("[3/4] 加载训练数据 ...")
    train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    val_dataset = None
    if Path(val_file).exists() and not args.dry_run:
        val_dataset = load_dataset("json", data_files=str(val_file), split="train")
        print(f"  train={len(train_dataset)}, val={len(val_dataset)}")
    else:
        print(f"  train={len(train_dataset)}, val=<disabled>")

    # ChatML 格式化：Unsloth packing 要求返回 list[str]（支持 batched map）
    def formatting_func(examples):
        msgs = examples["messages"]
        # 单条: messages 是 [{role, content}, ...]
        # 批量: messages 是 [[{...}, ...], [{...}, ...], ...]
        if msgs and isinstance(msgs[0], dict):
            return [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                )
            ]
        return [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False,
            )
            for m in msgs
        ]

    # Step 4: 配置 SFTTrainer（trl>=0.12 用 SFTConfig）
    print("[4/4] 配置 SFTTrainer ...")
    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        optim=OPTIM,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        save_strategy="steps" if not args.dry_run else "no",
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if (val_dataset is not None and not args.dry_run) else "no",
        eval_steps=eval_steps,
        seed=SEED,
        output_dir=str(args.output_dir),
        logging_dir=str(args.logs_dir),
        report_to="none",
        load_best_model_at_end=False,
        save_safetensors=True,
        # SFT 专属参数（勿再放进 TrainingArguments）
        dataset_num_proc=DATASET_NUM_PROC,
        packing=PACKING,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_func,
    )
    print("  [INFO] assistant_only_loss=True 启用")

    # 断点续训
    if args.resume_from:
        print(f"[INFO] 从 {args.resume_from} 续训")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # 保存 LoRA
    print("\n[SAVE] 保存 LoRA 适配器 ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  → {args.output_dir}")

    print("\n[GPU STATS]")
    gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    if gpu_stats:
        print(f"  GPU: {gpu_stats.name}, total_memory={gpu_stats.total_memory / 1e9:.2f} GB")
    if torch.cuda.is_available():
        print(f"  reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n✓ 训练完成")


if __name__ == "__main__":
    main()
