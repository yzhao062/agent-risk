"""Do token ids name the same strings across the four models' tokenizers?"""
import glob, json, os
from tokenizers import Tokenizer
B = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
names = {"qwen3_14b_fp8": "Qwen--Qwen3-14B-FP8", "qwen3_14b": "Qwen--Qwen3-14B",
         "qwen25_14b": "Qwen--Qwen2.5-14B-Instruct", "qwen35_27b_fp8": "Qwen--Qwen3.5-27B-FP8"}
tok = {}
for k, v in names.items():
    snaps = glob.glob(f"{B}/models--{v}/snapshots/*/tokenizer.json")
    if snaps:
        tok[k] = Tokenizer.from_file(snaps[0])
if "qwen3_14b_fp8" in tok:
    ref = tok["qwen3_14b_fp8"]
    rv = ref.get_vocab(with_added_tokens=True)
    inv_ref = {i: s for s, i in rv.items()}
    out = {}
    for k, t in tok.items():
        v = t.get_vocab(with_added_tokens=True)
        inv = {i: s for s, i in v.items()}
        shared = [i for i in inv_ref if i in inv]
        same = sum(1 for i in shared if inv[i] == inv_ref[i])
        out[k] = {"vocab_entries": len(v), "max_id": max(inv), "ids_in_both": len(shared), "same_string_at_same_id": same,
                  "first_differing_ids": sorted(i for i in shared if inv[i] != inv_ref[i])[:8]}
    txt = "I'll create the solution.sh script.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>"
    out["sample_ids"] = {k: t.encode(txt).ids[:24] for k, t in tok.items()}
    print(json.dumps(out, indent=1))
else:
    print(f"Tokenizers not found under {B}; set HF_HUB_CACHE to the directory holding model snapshots.")
