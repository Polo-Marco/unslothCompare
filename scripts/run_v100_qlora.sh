#!/usr/bin/env bash
set -euo pipefail
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=true
# cache to speed up later runs
mkdir -p ~/.triton && export TRITON_CACHE_DIR=~/.triton

python train_compare.py \
  --framework unsloth \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_type qlora \
  --gpu v100 \
  --precision fp32 \
  --quant none \
  --seq_len 4096 \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 4 \
  --dataset_path yentinglin/TaiwanChat \
  --dataset_split "train[:50]" \
  --sample_size 50 \
  --export_vllm_dir exports/qwen25_3b_qlora_v100_smoke \
  --log_json results/unsloth_qwen25_3b_qlora_v100_smoke.json

cat results/unsloth_qwen25_3b_qlora_v100_smoke.json
echo "Process Finished"
