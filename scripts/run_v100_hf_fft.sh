# text-only niceties
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=true
mkdir -p ~/.triton && export TRITON_CACHE_DIR=~/.triton

# HF LoRA baseline (fp16, no quant)
python train_compare.py \
  --framework hf \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_type lora \
  --gpu v100 \
  --precision fp32 \
  --quant none \
  --seq_len 1024 \
  --epochs 1 \
  --bsz 1 \
  --grad_accum 4 \
  --dataset_path yentinglin/TaiwanChat \
  --dataset_split "train[:50]" \
  --sample_size 50 \
  --export_vllm_dir exports/hf_qwen25_3b_lora_v100_smoke \
  --log_json results/hf_qwen25_3b_lora_v100_smoke.json

echo results/hf_qwen25_3b_lora_v100_smoke.json

