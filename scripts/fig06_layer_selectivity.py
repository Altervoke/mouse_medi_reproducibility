"""Figure 6: MEDI-set stimulus selectivity by cortical layer."""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_figure_style import apply, HEATMAP_CMAP
apply()
plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
areas, layers = ["V1", "LM", "RL", "AL"], ["L2/3", "L4", "L5"]
d = pd.read_csv(ROOT / "figures" / "tables" / "layer_selectivity_summary.csv")
if len(d) != 12 or d[["area", "layer"]].duplicated().any():
    raise ValueError("layer_selectivity_summary.csv must contain one row for each 4 x 3 area-layer cell")
v = d.pivot(index="area", columns="layer", values="sparseness").reindex(index=areas, columns=layers)
n = d.pivot(index="area", columns="layer", values="n").reindex(index=areas, columns=layers)
ann = v.copy().astype(object)
for area in areas:
    for layer in layers:
        ann.loc[area, layer] = f"{v.loc[area, layer]:.3f}\n(n={int(n.loc[area, layer]):,})"
fig, ax = plt.subplots(figsize=(4.25, 3.1))
sns.heatmap(v, annot=ann, fmt="", cmap=HEATMAP_CMAP, vmin=.2, vmax=.45, linewidths=.6, linecolor="white", annot_kws={"fontsize": 13}, cbar_kws={"label": "Median MEDI-set stimulus selectivity"}, ax=ax)
ax.set_xlabel("Cortical layer"); ax.set_ylabel("Visual area")
fig.tight_layout(); fig.savefig(ROOT / "figures/pdf/fig06_layer_selectivity.pdf"); plt.close(fig)
