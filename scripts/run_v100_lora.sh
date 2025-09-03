#!/usr/bin/env bash
set -euo pipefail
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=true
# cache to speed up later runs
mkdir -p ~/.triton && export TRITON_CACHE_DIR=~/.triton

python train_compare.py \
  --framework unsloth \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_type lora \
  --gpu v100 \
  --precision fp16 \
  --quant none \
  --seq_len 1024 \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 4 \
  --dataset_path yentinglin/TaiwanChat \
  --dataset_split "train[:500]" \
  --sample_size 500 \
  --export_vllm_dir exports/qwen25_7b_lora_v100_smoke \
  --log_json results/unsloth_qwen25_7b_lora_v100_smoke.json

