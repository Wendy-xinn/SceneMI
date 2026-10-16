#!/usr/bin/env python3
"""Plot the saved metric snapshot without querying or modifying training."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "TREND_REVIEW.json").read_text())
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
palette = {"clean": "#1769aa", "noisy": "#d16b0c"}
train = data["train_1000_update_blocks"]
full = [r for r in train if r["complete_1000_update_block"]]
partial = [r for r in train if not r["complete_1000_update_block"]]
ax = axes[0, 0]
ax.plot([r["last_step"] / 1000 for r in full], [r["mean_total_loss"] for r in full],
        "o-", color="#6a3d9a", label="Mean per 1,000 updates")
if partial:
    r = partial[-1]
    prev = full[-1]
    ax.plot([prev["last_step"] / 1000, r["last_step"] / 1000],
            [prev["mean_total_loss"], r["mean_total_loss"]], "--", color="#6a3d9a")
    ax.scatter([r["last_step"] / 1000], [r["mean_total_loss"]], marker="D", s=60,
               facecolors="white", edgecolors="#6a3d9a",
               label=f"Partial block: {r['n']} updates")
ax.set_title("Total training loss (weighted objective)")
ax.set_ylabel("Loss")
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)

specs = [
    (axes[0, 1], "mpjpe_mm", "Fixed VAL: joint position error", "MPJPE (mm)", "validation_summary"),
    (axes[1, 0], "root_orientation_deg", "Fixed VAL: root orientation error", "Error (degrees)", "validation_summary"),
    (axes[1, 1], "body_local_foot_error_mm", "Fixed VAL: body-local foot error", "Error (mm)", "foot_monitor_summary"),
]
for ax, metric, title, ylabel, source in specs:
    for regime in ("clean", "noisy"):
        tag = f"fixed/ego/{regime}/scene_on" if source == "validation_summary" else f"fixed/{regime}"
        rows = [r for r in data[source] if r["panel_regime"] == tag]
        xs = [r["step"] / 1000 for r in rows]
        ys = [r["metrics"][metric]["value"] if source == "validation_summary" else r["metrics"][metric] for r in rows]
        ax.plot(xs, ys, "o-", color=palette[regime], label=regime.capitalize(), markersize=4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

for ax in axes.flat:
    ax.set_xlabel("Training updates (thousands)")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=0)

fig.suptitle("Current baseline: training loss falls, but fixed VAL position error rebounds at 9k", fontsize=15)
fig.supxlabel("Fixed VAL = 3 matched windows, one per dataset; same input / sampling seeds. Not full VAL.\n"
               "Pose / orientation: count-weighted; local feet: equal-window mean. All panels: lower is better.",
               fontsize=10)
fig.savefig(HERE / "TRAINING_TRENDS.png", dpi=170, facecolor="white")
print(HERE / "TRAINING_TRENDS.png")
