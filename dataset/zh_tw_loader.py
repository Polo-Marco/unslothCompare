from datasets import load_dataset, Dataset
from typing import Optional, List, Dict

# normalize multiple schemas to a single 'text' field in zh-TW
def _to_dialog_text(ex: Dict) -> str:
    # Instruction/Output style
    if "instruction" in ex and ("output" in ex or "response" in ex):
        out = ex.get("output", ex.get("response", ""))
        instr = ex["instruction"]
        return f"使用者：{instr}\n助理：{out}"
    # conversations list
    if "conversations" in ex and isinstance(ex["conversations"], list):
        turns = []
        for turn in ex["conversations"]:
            role = turn.get("from","user")
            content = turn.get("value","")
            role_zh = "使用者" if role in ["user","human"] else "助理"
            turns.append(f"{role_zh}：{content}")
        return "\n".join(turns)
    # single text column
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"]
    # fallback: join all string fields
    fields = [str(v) for k,v in ex.items() if isinstance(v, str)]
    return "\n".join(fields)

def load_zh_tw_sft(tokenizer, name_or_path: str, max_len: int, split: str="train", sample_size: Optional[int]=None) -> Dataset:
    """
    name_or_path: HF Hub dataset name or local path
    Returns a HF Dataset with a single 'text' column
    """
    ds = load_dataset(name_or_path, split=split)
    ds = ds.map(lambda ex: {"text": _to_dialog_text(ex)},
                remove_columns=[c for c in ds.column_names if c != "text"])
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
    return ds

