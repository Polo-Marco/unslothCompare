import os, torch

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cuda_mem_snapshot():
    if not torch.cuda.is_available():
        return 0.0
    gb = torch.cuda.max_memory_allocated() / (1024**3)
    return round(gb, 3)

def save_for_vllm(model, tokenizer, out_dir: str, prefer_bf16: bool = False):
    """
    Merge LoRA (if any), cast to fp16/bf16, and save a vanilla HF folder that vLLM can load.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Try merge-and-unload if it's a PEFT-wrapped model
    try:
        model = model.merge_and_unload()
        print("[export] merged LoRA adapters into base weights.")
    except Exception:
        print("[export] no PEFT adapters to merge (probably full FT).")

    # Ensure dtype friendly to vLLM
    target_dtype = torch.bfloat16 if prefer_bf16 and torch.cuda.is_available() else torch.float16
    try:
        model.to(target_dtype)
    except Exception:
        pass

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[export] vLLM-ready model saved -> {out_dir}")

