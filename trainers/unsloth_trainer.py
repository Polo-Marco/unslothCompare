import os, time, math
import torch
from dataclasses import dataclass
from typing import Dict, Any

from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

from unsloth import FastLanguageModel
from datasets import Dataset

from trainers.utils import set_seed, cuda_mem_snapshot, save_for_vllm
from dataset.zh_tw_loader import load_zh_tw_sft

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _make_model_and_tok(args):
    compute_dtype = DTYPE_MAP[args.precision]
    load_in_4bit = args.quant == "int4"
    load_in_8bit = args.quant == "int8"

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.seq_len,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.train_type in ["lora", "qlora"]:
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",  # optional but recommended
        )
        return model, tok

def run_unsloth(args) -> Dict[str, Any]:
    """
    Manual SFT loop to avoid accelerate/TRL optimizer edge cases.
    """
    set_seed(42)
    model, tok = _make_model_and_tok(args)
    ds = load_zh_tw_sft(tok, args.dataset_path, args.seq_len, split=args.dataset_split, sample_size=args.sample_size)

    # Tokenize once; tiny collator
    def tokenize_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len, padding="max_length")

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # DataLoader
    from torch.utils.data import DataLoader
    dl = DataLoader(ds_tok, batch_size=args.bsz, shuffle=True, drop_last=True)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    model.train()
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    steps = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attn = batch["attention_mask"].to(model.device, non_blocking=True)
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            (out.loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1

    wall = time.time() - start
    peak_vram = cuda_mem_snapshot()

    # rough tokens/sec estimate
    tokens_per_step = args.bsz * args.grad_accum * args.seq_len
    total_steps = steps
    tokens_total = tokens_per_step * total_steps
    tps = tokens_total / wall if wall > 0 else 0.0

    # Save adapters/base
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Export merged for vLLM
    merged_path = args.export_vllm_dir
    save_for_vllm(model, tok, merged_path, prefer_bf16=(args.gpu=="h100"))

    metrics = {
        "tokens_per_sec_est": round(tps, 2),
        "peak_vram_gb": peak_vram,
        "wall_clock_sec": round(wall, 2),
        "train_steps": total_steps,
        "config": {
            "model_name": args.model_name,
            "train_type": args.train_type,
            "gpu": args.gpu,
            "precision": args.precision,
            "quant": args.quant,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "bsz": args.bsz,
            "grad_accum": args.grad_accum,
        },
        "export_vllm_dir": merged_path,
        "output_dir": args.output_dir,
        "dataset": {"path": args.dataset_path, "split": args.dataset_split, "size": len(ds)},
    }
    print(f"[unsloth/manual] metrics: {metrics}")
    return metrics

