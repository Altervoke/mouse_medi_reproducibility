"""Figure S12: complete parameter-resolved selectivity--tolerance correlations."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_figure_style import AREA_COLORS, apply, panel
apply()
tables = ROOT / "figures/tables"
d = pd.read_csv(tables / "selectivity_tolerance_by_parameter.csv")
conditions = ["contrast", "rotation", "translation", "temporal_reverse", "speed_up", "slow_down"]
labels = {"temporal_reverse": "temporal reverse", "speed_up": "speed up", "slow_down": "slow down"}
xlabels = {"contrast": "contrast multiplier", "rotation": "rotation (deg)",
           "translation": "translation (px)", "temporal_reverse": "temporal reverse",
           "speed_up": "temporal speed multiplier", "slow_down": "temporal speed multiplier"}
areas = ["V1", "LM", "RL", "AL"]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1), sharey=True)
for i, (ax, condition) in enumerate(zip(axes.flat, conditions)):
    q = d[d.condition == condition]
    for area in areas:
        z = q[q.area == area].sort_values("parameter")
        ax.plot(z.parameter, z.rho, marker="o", ms=3.5, color=AREA_COLORS[area], label=area)
    ax.axhline(0, color="0.3", lw=.8)
    ax.set_title(labels.get(condition, condition)); ax.set_xlabel(xlabels[condition]); ax.set_ylabel("Spearman $\\rho$")
    values = sorted(q.parameter.unique()); ax.set_xticks(values)
    if condition == "temporal_reverse": ax.set_xticklabels(["-1"])
    elif condition in {"contrast", "speed_up", "slow_down"}: ax.set_xticklabels([f"{x:g}x" for x in values])
    else: ax.set_xticklabels([f"{x:g}" for x in values])
    if i < 3:
        ax.tick_params(axis="x", labelrotation=85 if condition == "rotation" else 35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    ax.grid(axis="y", color="0.88", linewidth=.6); panel(ax, chr(97 + i))
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.52, 1.0))
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(ROOT / "figures/pdf/fig14_selectivity_parameter_rho.pdf"); plt.close(fig)
