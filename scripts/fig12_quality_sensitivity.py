"""Generate the quality-threshold sensitivity figure (Figure S7)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from paper_figure_style import AREA_COLORS, apply, panel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "pdf"
TABLES = ROOT / "figures" / "tables"
QUALITY = TABLES / "quality_sensitivity_summary.csv"
AREAS = ["V1", "LM", "RL", "AL"]
ORDER = ["other", "near-static candidate", "response-invariant despite large change"]
DISPLAY = {
    "other": "other",
    "near-static candidate": "near-static",
    "response-invariant despite large change": "large change",
}

apply()

families = ["contrast", "rotation", "translation", "temporal_reverse", "speed_up", "slow_down"]
filters = [
    "all",
    "original_ge_q05",
    "original_ge_q10",
    "original_ge_q25",
    "cc_ge_0.3",
    "cc_ge_0.5",
    "cc_ge_0.7",
]
filter_labels = {
    "all": "all",
    "original_ge_q05": "resp.\nq05+",
    "original_ge_q10": "resp.\nq10+",
    "original_ge_q25": "resp.\nq25+",
    "cc_ge_0.3": "cc\n.3+",
    "cc_ge_0.5": "cc\n.5+",
    "cc_ge_0.7": "cc\n.7+",
}
quality = pd.read_csv(QUALITY)
quality["filter_label"] = pd.Categorical(quality["filter"].map(filter_labels), categories=[filter_labels[x] for x in filters], ordered=True)
fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4), sharey=False)
for i, (ax, family) in enumerate(zip(axes.flat, families)):
    q = quality[quality.family.eq(family)]
    sns.lineplot(
        data=q,
        x="filter_label",
        y="contrast_vs_v1",
        hue="brain_area",
        hue_order=["LM", "RL", "AL"],
        marker="o",
        linewidth=1.2,
        markersize=4,
        palette=AREA_COLORS,
        ax=ax,
    )
    for area in ["LM", "RL", "AL"]:
        z = q[q.brain_area.eq(area)].sort_values("filter_label")
        ax.errorbar(range(len(z)), z.contrast_vs_v1, yerr=[z.contrast_vs_v1-z.ci_low, z.ci_high-z.contrast_vs_v1], fmt='none', ecolor=AREA_COLORS[area], elinewidth=.6, capsize=1.5, alpha=.8)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title(family.replace("_", " "))
    ax.set_xlabel("readout-quality subset")
    ax.set_ylabel("median retention difference")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.grid(axis="y", color="0.88", linewidth=.6)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    panel(ax, chr(97 + i))
handles = [plt.Line2D([], [], color=AREA_COLORS[a], marker='o', label=a) for a in ["LM", "RL", "AL"]]
fig.legend(handles=handles, title="Area", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.52, 1.0))
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUT / "fig12_quality_sensitivity.pdf")
plt.close(fig)
