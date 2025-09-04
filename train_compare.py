# ./train_compare.py
import argparse, os, json, time
from dataclasses import dataclass

def build_args():
    ap = argparse.ArgumentParser()
    # experiment axes
    ap.add_argument("--framework", choices=["unsloth","hf"], default="unsloth")
    ap.add_argument("--model_name", required=True, help="e.g. meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--train_type", choices=["full", "lora", "qlora"], required=True)
    ap.add_argument("--gpu", choices=["v100", "h100"], required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default=None)
    # data
    ap.add_argument("--dataset_path", required=True, help="HF Hub name or local path")
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--sample_size", type=int, default=None, help="for quick smoke tests")
    ap.add_argument("--seq_len", type=int, default=4096)
    # training hparams
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--eval_ratio", type=float, default=0.1)
    # output / logging
    ap.add_argument("--output_dir", default="out/unsloth_run")
    ap.add_argument("--export_vllm_dir", default="exports/unsloth_merged_vllm")
    ap.add_argument("--log_json", default="results/unsloth_metrics.json")
    return ap.parse_args()

def resolve_policy(args):
    # precision default
    if args.precision is None:
        args.precision = "fp16" if args.gpu == "v100" else "bf16"
    # quant default
    if args.quant is None:
        args.quant = "int4" if args.train_type == "qlora" else "none"
    # validations
    if args.gpu == "v100" and args.precision == "bf16":
        raise ValueError("V100 does not support bf16; use fp16.")
    if args.train_type == "full" and args.quant != "none":
        raise ValueError("Full FT should not use int8/int4 weight loading.")
    if args.train_type == "lora" and args.quant not in ["none", "int8", "int4"]:
        raise ValueError("LoRA quant must be none/int8/int4.")

def main():
    args = build_args()
    print(f"args:\n {args}")
    resolve_policy(args)
    os.makedirs(os.path.dirname(args.log_json), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.export_vllm_dir, exist_ok=True)
    if args.framework == "unsloth":
        from trainers.unsloth_trainer import run_unsloth
        metrics = run_unsloth(args)
    else:
        from trainers.hf_trainer import run_hf
        metrics = run_hf(args)   
    with open(args.log_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] metrics saved -> {args.log_json}")

if __name__ == "__main__":
    main()

