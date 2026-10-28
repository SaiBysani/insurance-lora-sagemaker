"""Figures for the newsletter that do NOT depend on the SageMaker run.

1. sagemaker_workflow.png   - conceptual workflow diagram (not a screenshot)
2. dataset_composition.png  - counts measured from train.jsonl / test.jsonl

Run from the project folder:  python AWS_FINAL/make_figures.py
"""
import collections
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def load(fn):
    with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
        return [json.loads(l)["messages"][-1]["content"] for l in f]


# ---------------------------------------------------------------- dataset chart
train = [json.loads(a) for a in load("train.jsonl")]
test = [json.loads(a) for a in load("test.jsonl")]
decisions = ["covered", "excluded", "covered_by_exception", "outside_policy_period"]
labels = {"covered": "Covered", "excluded": "Excluded",
          "covered_by_exception": "Covered by exception", "outside_policy_period": "Outside policy period"}
types = ["commercial_property", "homeowners", "commercial_auto", "cyber_liability"]
tlabels = {"commercial_property": "Commercial property", "homeowners": "Homeowners",
           "commercial_auto": "Commercial auto", "cyber_liability": "Cyber liability"}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
for ax, key, order, lab, title in [
    (axes[0], lambda t: t["claim"]["decision"], decisions, labels, "Claim decision"),
    (axes[1], lambda t: t["policy_type"], types, tlabels, "Line of business"),
]:
    c = collections.Counter(key(t) for t in test)
    vals = [c[k] for k in order]
    y = range(len(order))
    ax.set_facecolor(SURFACE)
    ax.barh(list(y), vals, height=0.55, color=BLUE)
    for yy, v in zip(y, vals):
        ax.text(v + 1.5, yy, str(v), va="center", color=INK, fontsize=10)
    ax.set_yticks(list(y)); ax.set_yticklabels([lab[k] for k in order]); ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.2); ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
fig.suptitle("The 150 held-out test policies (synthetic data, seed 42)", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.005, f"Training set: {len(train)} policies. Counts read directly from test.jsonl. "
         "Not a model result.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(os.path.join(HERE, "dataset_composition.png"), dpi=170, facecolor=SURFACE)
plt.close(fig)

# --------------------------------------------------------------- workflow diagram
fig, ax = plt.subplots(figsize=(13.5, 6.4), facecolor=SURFACE)
ax.set_xlim(0, 134); ax.set_ylim(0, 64); ax.axis("off")


def box(x, y, w, h, title, body, edge=INK2, fill="#ffffff", tcolor=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + 1.4, y + h - 2.2, title, fontsize=10, fontweight="bold", color=tcolor, va="top")
    ax.text(x + 1.4, y + h - 6.2, body, fontsize=8.6, color=INK2, va="top", linespacing=1.45)


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=INK2, lw=1.4, ls=ls, shrinkA=0, shrinkB=0))


ax.text(1, 62, "Fine-tuning workflow on Amazon SageMaker AI", fontsize=15, fontweight="bold", color=INK, va="top")
ax.text(1, 58.2, "Conceptual diagram of how this project ran. Not a console screenshot.",
        fontsize=9, color=MUTED, va="top")

# Training job container
ax.add_patch(FancyBboxPatch((25, 6), 107, 46, boxstyle="round,pad=0.4,rounding_size=2",
                            fc="#fff6ef", ec=ORANGE, lw=1.6, ls="--"))
ax.text(26.5, 50.6, "SageMaker AI training job on 1x ml.g5.2xlarge (NVIDIA A10G, 24 GB), billed per second",
        fontsize=9.5, fontweight="bold", color="#a8461c", va="top")

box(1, 30, 21, 20, "1  Synthetic data",
    "insurance_data.py\n800 train / 150 test\nheld-out wordings\nSFT \"messages\" JSONL")
box(28, 30, 23.5, 16, "2  Load base model",
    "Qwen3-4B-Instruct-2507\nfrom Hugging Face\nbf16, one GPU")
box(54.5, 30, 23.5, 16, "3  Baseline eval",
    "150 test policies\ngreedy decoding\nauto-graded JSON")
box(81, 30, 23.5, 16, "4  SFT + LoRA",
    "rank 16, alpha 32\nall attn + MLP proj.\n2 epochs, lr 2e-4\nloss on JSON only")
box(107.5, 30, 23, 16, "5  Fine-tuned eval",
    "same 150 policies\nsame prompt + scorer\n+ 30 x 3 sampled\nconsistency runs")
box(54.5, 9, 36, 15, "6  Outputs",
    "Synced to Amazon S3 as the job runs:\nresults.json, every prediction,\nloss history, LoRA adapter")
box(95, 9, 35.5, 15, "7  Job ends",
    "SageMaker AI tears the instance\ndown when the script finishes.\nNothing is left running.")

arrow(22.4, 40, 27.6, 40); arrow(51.9, 38, 54.1, 38); arrow(78.4, 38, 80.6, 38); arrow(104.9, 38, 107.1, 38)
arrow(119, 29.6, 80, 24.4); arrow(90.9, 16.5, 94.6, 16.5)

# Alternative path
ax.add_patch(FancyBboxPatch((1, 6), 21, 18, boxstyle="round,pad=0.4,rounding_size=1.6",
                            fc=SURFACE, ec=MUTED, lw=1.2, ls=":"))
ax.text(2.4, 21.8, "No-code option", fontsize=10, fontweight="bold", color=INK2, va="top")
ax.text(2.4, 17.8, "The same train.jsonl also\nworks in SageMaker AI\nserverless model\ncustomization (not used\nin this project)",
        fontsize=8.4, color=MUTED, va="top", linespacing=1.45)
arrow(11.5, 29.6, 11.5, 24.6, ls=":")

fig.savefig(os.path.join(HERE, "sagemaker_workflow.png"), dpi=170, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("wrote dataset_composition.png and sagemaker_workflow.png")
