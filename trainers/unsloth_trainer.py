# ./trainers/unsloth_trainer.py
import os, time, math
import torch
from typing import Dict, Any

from transformers import AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel

from trainers.utils import set_seed, cuda_mem_snapshot, save_for_vllm
from dataset.zh_tw_loader import load_zh_tw_sft

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _make_model_and_tok(args):
    compute_dtype = DTYPE_MAP[args.precision]

    # prefer BitsAndBytesConfig over deprecated load_in_* flags
    bnb_cfg = None
    if args.quant in ("int8", "int4"):
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit = (args.quant == "int8"),
            load_in_4bit = (args.quant == "int4"),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.seq_len,
        dtype=compute_dtype,
        quantization_config=bnb_cfg,      # <¡X new
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
            use_gradient_checkpointing="unsloth",
        )
    return model, tok


def run_unsloth(args) -> Dict[str, Any]:
    """
    Manual SFT loop with metrics: training time, train loss, eval loss.
    """
    set_seed(42)
    model, tok = _make_model_and_tok(args)

    # ---- load & split dataset
    ds = load_zh_tw_sft(tok, args.dataset_path, args.seq_len,
                        split=args.dataset_split, sample_size=args.sample_size)
    eval_ratio = getattr(args, "eval_ratio", 0.1)
    if eval_ratio > 0 and len(ds) > 1:
        splits = ds.train_test_split(test_size=eval_ratio, seed=42, shuffle=True)
        ds_train = splits["train"]
        ds_eval  = splits["test"]
    else:
        ds_train = ds
        ds_eval  = None

    # ---- tokenize (batched)
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len, padding="max_length")

    ds_train_tok = ds_train.map(tok_fn, batched=True, remove_columns=[c for c in ds_train.column_names if c != "text"])
    ds_train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if ds_eval is not None:
        ds_eval_tok = ds_eval.map(tok_fn, batched=True, remove_columns=[c for c in ds_eval.column_names if c != "text"])
        ds_eval_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        ds_eval_tok = None

    # ---- dataloaders
    from torch.utils.data import DataLoader
    dl_train = DataLoader(ds_train_tok, batch_size=args.bsz, shuffle=True, drop_last=True)
    dl_eval  = DataLoader(ds_eval_tok,  batch_size=args.bsz, shuffle=False, drop_last=False) if ds_eval_tok else None

    # ---- optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- warm-up forward (JIT compile) to avoid "stall" on first step
    print("[stage] warm-up forward (JIT compile kernels)...")
    with torch.inference_mode():
        warm_ids = torch.full((1, min(256, args.seq_len)), tok.pad_token_id, device=model.device)
        _ = model(input_ids=warm_ids, attention_mask=torch.ones_like(warm_ids))
    print("[stage] warm-up done.")

    # ---- train
    model.train()
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_steps = 0
    running_loss = 0.0
    n_updates = 0
    last_step_loss = None
    tokens_seen = 0

    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for i, batch in enumerate(dl_train):
            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attn = batch["attention_mask"].to(model.device, non_blocking=True)
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            loss = out.loss
            (loss / args.grad_accum).backward()

            # update counters
            tokens_seen += int(input_ids.numel())
            last_step_loss = float(loss.detach().cpu())

            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                total_steps += 1
                running_loss += last_step_loss
                n_updates += 1

    wall = time.time() - start
    peak_vram = cuda_mem_snapshot()

    # ---- metrics (train)
    train_loss_mean = (running_loss / max(1, n_updates)) if n_updates else (last_step_loss or 0.0)
    ex_per_sec = (len(ds_train_tok)) / wall if wall > 0 else 0.0
    tokens_per_sec = tokens_seen / wall if wall > 0 else 0.0

    # ---- eval (loss + ppl)
    eval_loss = None
    eval_ppl  = None
    if dl_eval is not None:
        model.eval()
        with torch.no_grad():
            total, count = 0.0, 0
            for batch in dl_eval:
                input_ids = batch["input_ids"].to(model.device, non_blocking=True)
                attn = batch["attention_mask"].to(model.device, non_blocking=True)
                out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
                total += float(out.loss.detach().cpu())
                count += 1
        if count:
            eval_loss = total / count
            try:
                eval_ppl = math.exp(eval_loss)
            except OverflowError:
                eval_ppl = float("inf")

    # ---- save + export
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    merged_path = args.export_vllm_dir
    save_for_vllm(model, tok, merged_path, prefer_bf16=(args.gpu=="h100"))

    # ---- final metrics package
    metrics = {
        "wall_clock_sec": round(wall, 2),
        "examples_per_sec": round(ex_per_sec, 2),
        "tokens_per_sec_est": round(tokens_per_sec, 2),
        "peak_vram_gb": peak_vram,
        "train_steps": total_steps,
        "train_loss_mean": round(train_loss_mean, 5) if train_loss_mean is not None else None,
        "train_loss_last": round(last_step_loss, 5) if last_step_loss is not None else None,
        "eval_loss": round(eval_loss, 5) if eval_loss is not None else None,
        "eval_ppl": round(eval_ppl, 5) if eval_ppl is not None else None,
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
            "eval_ratio": eval_ratio,
        },
        "export_vllm_dir": merged_path,
        "output_dir": args.output_dir,
        "dataset": {
            "path": args.dataset_path,
            "split": args.dataset_split,
            "size_total": len(ds),
            "size_train": len(ds_train_tok),
            "size_eval": len(ds_eval_tok) if ds_eval_tok else 0,
        },
    }
    print(f"[unsloth/manual] metrics: {metrics}")
    return metrics


