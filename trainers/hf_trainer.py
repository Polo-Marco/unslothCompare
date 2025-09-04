# trainers/hf_trainer.py
import os, time, math
from typing import Dict, Any, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trainers.utils import set_seed, cuda_mem_snapshot
# we keep save_for_vllm import if you use it here; else remove
from trainers.utils import save_for_vllm
from dataset.zh_tw_loader import load_zh_tw_sft

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def _build_bnb_config(args):
    if args.quant not in ("int8", "int4"):
        return None
    return BitsAndBytesConfig(
        load_in_8bit=(args.quant == "int8"),
        load_in_4bit=(args.quant == "int4"),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPE_MAP[args.precision],
    )

def _make_model_and_tok(args):
    compute_dtype = DTYPE_MAP[args.precision]
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = _build_bnb_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=None if quant_cfg is not None else compute_dtype,
    )

    if args.train_type in ["lora", "qlora"]:
        lconf = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        )
        model = get_peft_model(model, lconf)

    return model, tok

def _tokenize(tok, seq_len: int):
    def _fn(batch):
        return tok(batch["text"], truncation=True, max_length=seq_len, padding="max_length")
    return _fn

def _avg_losses_from_log(log_history: List[dict]):
    losses = [l["loss"] for l in log_history if "loss" in l]
    last = losses[-1] if losses else None
    mean = sum(losses)/len(losses) if losses else None
    return mean, last

def run_hf(args) -> Dict[str, Any]:
    set_seed(42)
    model, tok = _make_model_and_tok(args)

    # ----- data
    ds = load_zh_tw_sft(tok, args.dataset_path, args.seq_len,
                        split=args.dataset_split, sample_size=args.sample_size)
    eval_ratio = getattr(args, "eval_ratio", 0.1)
    if eval_ratio > 0 and len(ds) > 1:
        sp = ds.train_test_split(test_size=eval_ratio, seed=42, shuffle=True)
        ds_train, ds_eval = sp["train"], sp["test"]
    else:
        ds_train, ds_eval = ds, None

    tok_fn = _tokenize(tok, args.seq_len)
    ds_train_tok = ds_train.map(tok_fn, batched=True, remove_columns=[c for c in ds_train.column_names if c != "text"])
    ds_train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    ds_eval_tok = None
    if ds_eval is not None:
        ds_eval_tok = ds_eval.map(tok_fn, batched=True, remove_columns=[c for c in ds_eval.column_names if c != "text"])
        ds_eval_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # ----- precision policy (pure FP16 on V100 to avoid GradScaler unscale error)
    want_fp16 = (args.precision == "fp16")
    want_bf16 = (args.precision == "bf16")
    force_no_scaler = (args.gpu.lower() == "v100" and want_fp16) or (os.getenv("HF_FORCE_NO_SCALER", "0") == "1")
    if force_no_scaler:
        print("[hf] Using pure FP16 (no GradScaler) to avoid unscale() errors on V100.")
        try:
            model = model.to(dtype=torch.float16)
        except Exception as e:
            print(f"[hf] warning: model.half() failed with: {e}")

    do_eval = ds_eval_tok is not None

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=0,
        report_to=[],
        # ---- avoid HF eval path; we'll run manual eval below
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        optim="adamw_torch",
        fp16=(want_fp16 and not force_no_scaler),
        bf16=(want_bf16 and not force_no_scaler),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        processing_class=tok,   # replaces deprecated tokenizer=
        data_collator=collator,
        train_dataset=ds_train_tok,
        eval_dataset=None,      # we¡¦ll eval manually
    )

    # ----- train
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    wall = time.time() - start
    peak_vram = cuda_mem_snapshot()

    global_steps = int(trainer.state.global_step or 0)
    train_mean, train_last = _avg_losses_from_log(trainer.state.log_history)

    tokens_per_step = args.bsz * args.grad_accum * args.seq_len
    tps = (tokens_per_step * global_steps) / wall if wall > 0 else 0.0
    ex_per_sec = len(ds_train_tok) / wall if wall > 0 else 0.0

    # ----- manual eval (loss + ppl)
    eval_loss = None
    eval_ppl  = None
    if do_eval:
        from torch.utils.data import DataLoader
        dl_eval = DataLoader(ds_eval_tok, batch_size=args.bsz, shuffle=False, drop_last=False, collate_fn=collator)
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in dl_eval:
                for k in ("input_ids", "attention_mask", "labels"):
                    batch[k] = batch[k].to(model.device, non_blocking=True)
                out = model(**batch)
                total += float(out.loss.detach().cpu())
                count += 1
        if count:
            eval_loss = total / count
            try:
                eval_ppl = math.exp(eval_loss)
            except OverflowError:
                eval_ppl = float("inf")

    # ----- save / export (merge LoRA if present)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    merged_path = args.export_vllm_dir
    try:
        _merged = trainer.model.merge_and_unload()
        _merged.save_pretrained(merged_path); tok.save_pretrained(merged_path)
        print("[export] merged LoRA adapters into base weights (HF).")
    except Exception:
        trainer.model.save_pretrained(merged_path); tok.save_pretrained(merged_path)
        print("[export] saved base model (no adapters to merge).")

    metrics = {
        "wall_clock_sec": round(wall, 2),
        "examples_per_sec": round(ex_per_sec, 2),
        "tokens_per_sec_est": round(tps, 2),
        "peak_vram_gb": peak_vram,
        "train_steps": global_steps,
        "train_loss_mean": round(train_mean, 5) if train_mean is not None else None,
        "train_loss_last": round(train_last, 5) if train_last is not None else None,
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
            "size_eval": len(ds_eval_tok) if do_eval else 0,
        },
    }
    print(f"[hf/trainer] metrics: {metrics}")
    return metrics
