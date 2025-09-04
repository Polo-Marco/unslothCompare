#!/usr/bin/env bash
# This script runs different fine-tuning methods (full, lora, qlora)
# with various precision and quantization settings on a V100 GPU.

set -euo pipefail
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=true
# Cache to speed up later runs
mkdir -p ~/.triton && export TRITON_CACHE_DIR=~/.triton

# --- Configuration ---
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH="yentinglin/TaiwanChat"
GPU="v100"

# --- Training Type Loop ---
TRAIN_TYPES=("full" "lora" "qlora")
QUANTS=("none")
for train_type in "${TRAIN_TYPES[@]}"; do
    echo "--------------------------------------------------------"
    echo "Starting test for fine-tuning type: ${train_type}"
    echo "--------------------------------------------------------"

    case "${train_type}" in
        # Full fine-tuning requires significant VRAM. On a V100 GPU with a 3B model,
        # it is generally recommended to use fp16 to avoid Out-of-Memory (OOM) errors.
        "full")
            PRECISIONS=("fp16")
            ;;

        # LoRA is a parameter-efficient fine-tuning method. It can be run with
        # different precisions and quantizations for comparison.
        "lora")
            PRECISIONS=("fp32" "fp16")
            ;;

        # QLoRA specifically uses 4-bit quantization, which is automatically handled
        # by Unsloth when the train_type is set to 'qlora'.
        "qlora")
            PRECISIONS=("fp16") # QLoRA typically runs on fp16
            QUANTS=("none")    # The 'qlora' train_type handles 4-bit quantization
            ;;

        *)
            echo "Error: Unknown training type ${train_type}. Skipping."
            continue
            ;;
    esac

    for precision in "${PRECISIONS[@]}"; do
        for quant in "${QUANTS[@]}"; do
            # Create a unique name for the log and export directory
            export_dir="exports/unsloth_${MODEL_NAME//\//-}_${train_type}_${GPU}_${precision}_${quant}"
            log_file="results/unsloth_${MODEL_NAME//\//-}_${train_type}_${GPU}_${precision}_${quant}.json"

            echo "--- Running test: Train=${train_type}, Prec=${precision}, Quant=${quant} ---"
            echo "Exporting to: ${export_dir}"
            echo "Logging to: ${log_file}"

            python train_compare.py \
                --framework unsloth \
                --model_name "${MODEL_NAME}" \
                --train_type "${train_type}" \
                --gpu "${GPU}" \
                --precision "${precision}" \
                --quant "${quant}" \
                --seq_len 1024 \
                --epochs 5 \
                --bsz 1 \
                --grad_accum 4 \
                --dataset_path "${DATASET_PATH}" \
                --dataset_split "train[:2000]" \
                --sample_size 2000 \
                --output_dir "${export_dir}" \
                --log_json "${log_file}"

            # Display the results of the current test
            echo ""
            echo "Contents of ${log_file}:"
            cat "${log_file}"
            echo ""
        done
    done
done

echo "All V100 tests finished."
