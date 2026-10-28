"""Charts from the real SageMaker AI run.

Reads results/results.json (downloaded from the job's S3 checkpoint folder) and writes
base_vs_finetuned.png and loss_curve.png next to this script.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(os.path.dirname(HERE), "results", "results.json"), encoding="utf-8"))

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
BASE_C, FT_C = "#a3a29c", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def clean(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)


# ------------------------------------------------------------ base vs fine-tuned
rows = [("Clean JSON only", "strict_json"), ("Exact schema", "schema_ok"), ("Policy fields correct", "fields_acc"),
        ("Exclusions F1", "exclusions_f1"), ("Claim decision correct", "decision"),
        ("Payable amount correct", "payable"), ("Everything correct", "all_correct")]
base = [R["base"][k] for _, k in rows] + [R["consistency"]["base"]["identical_across_3_runs"]]
ft = [R["finetuned"][k] for _, k in rows] + [R["consistency"]["finetuned"]["identical_across_3_runs"]]
labels = [l for l, _ in rows] + ["Identical across 3 sampled runs*"]

y = np.arange(len(labels)); h = 0.36
fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
ax.barh(y - h / 2 - 0.02, base, h, color=BASE_C, label="Base model (prompt only)")
ax.barh(y + h / 2 + 0.02, ft, h, color=FT_C, label="Fine-tuned with LoRA")
for yy, v in zip(y - h / 2 - 0.02, base):
    ax.text(v + 1, yy, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
for yy, v in zip(y + h / 2 + 0.02, ft):
    ax.text(v + 1, yy, f"{v:.1f}%", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 112)
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); clean(ax)
ax.set_xlabel(f"% of {R['n_test']} held-out synthetic test policies")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.suptitle(f"Base vs LoRA fine-tuned {R['model_id'].split('/')[-1]}", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.01, "* 30 test policies x 3 runs at temperature 0.7. Measured on an Amazon SageMaker AI training job "
         "(ml.g5.2xlarge). Synthetic data.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(os.path.join(HERE, "base_vs_finetuned.png"), dpi=170, facecolor=SURFACE)
plt.close(fig)

# --------------------------------------------------------------------- loss curve
hist = R["loss_history"]
fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
ax.plot([p["step"] for p in hist], [p["loss"] for p in hist], color=FT_C, lw=2, marker="o", ms=4)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); clean(ax)
ax.set_xlabel("Optimizer step (effective batch 16)"); ax.set_ylabel("Training loss")
first, last = hist[0], hist[-1]
ax.annotate(f"{first['loss']:.4f}", (first["step"], first["loss"]), textcoords="offset points", xytext=(6, 2), fontsize=8.5, color=INK2)
ax.annotate(f"{last['loss']:.4f}", (last["step"], last["loss"]), textcoords="offset points", xytext=(-10, 8), fontsize=8.5, color=INK2)
fig.suptitle(f"LoRA training loss, {R['epochs']} epochs, {R['train_minutes']} min on one A10G", x=0.01, ha="left",
             fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.01, 0.01, "Loss on the JSON answer tokens only, logged every 5 steps. Measured on SageMaker AI.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(os.path.join(HERE, "loss_curve.png"), dpi=170, facecolor=SURFACE)
plt.close(fig)
print("wrote base_vs_finetuned.png and loss_curve.png")
