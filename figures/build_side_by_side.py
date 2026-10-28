"""Build side_by_side_example.md from a real notebook run.

After the SageMaker run, download the job outputs (base_predictions.jsonl, finetuned_predictions.jsonl,
results.json) into results/, then run:  python figures/build_side_by_side.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def load(fn):
    with open(os.path.join(OUT, fn), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


base, ft = load("base_predictions.jsonl"), load("finetuned_predictions.jsonl")
with open(os.path.join(os.path.dirname(HERE), "test.jsonl"), encoding="utf-8") as f:
    docs = [json.loads(l)["messages"][1]["content"] for l in f]

# First test policy (in file order) where the base model got the claim decision wrong and the
# fine-tuned model got everything right; falls back to any policy only the fine-tuned model got fully right.
flips = [i for i, (b, f) in enumerate(zip(base, ft)) if f["score"]["all_correct"] and not b["score"]["decision"]]
idx = flips[0] if flips else next((i for i, (b, f) in enumerate(zip(base, ft)) if f["score"]["all_correct"] and not b["score"]["all_correct"]), None)
if idx is None:
    raise SystemExit("No test policy where only the fine-tuned model was fully correct; nothing to show.")
print(f"{len(flips)} test policies where the base decision was wrong and the fine-tuned answer fully correct")

wrong = [k for k in ["valid_json", "strict_json", "schema_ok", "exclusions_f1", "claim_peril", "decision", "payable"]
         if base[idx]["score"][k] != ft[idx]["score"][k]]
md = [f"# Side-by-side example (test policy {idx})", "",
      "Real outputs from the SageMaker AI run. Synthetic policy.", "",
      "## Policy", "", "```text", docs[idx], "```", "",
      "## Base model answer", "", "```text", base[idx]["prediction"][:1500], "```", "",
      "## Fine-tuned (LoRA) answer", "", "```json", ft[idx]["prediction"], "```", "",
      "## Correct answer", "", "```json", json.dumps(base[idx]["gold"], indent=2), "```", "",
      "Score differences (base -> fine-tuned): " + ", ".join(f"{k}: {base[idx]['score'][k]} -> {ft[idx]['score'][k]}" for k in wrong)]
with open(os.path.join(OUT, "side_by_side_example.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("wrote side_by_side_example.md for test policy", idx)
