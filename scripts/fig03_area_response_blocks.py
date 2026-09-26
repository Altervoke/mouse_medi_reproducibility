"""Figure 3a: within-readout-normalized cross-area response blocks."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_figure_style import apply, HEATMAP_CMAP
apply()
table = pd.read_csv(ROOT / "figures/tables/within_readout_normalized_area_blocks.csv")
areas = ["V1", "LM", "RL", "AL"]
values = table.pivot(index="source_area", columns="target_area", values="median_within_readout_z").reindex(index=areas, columns=areas)
fig, ax = plt.subplots(figsize=(3.35, 3.0))
sns.heatmap(values, annot=True, fmt=".2f", center=0, cmap=HEATMAP_CMAP, linewidths=.5, linecolor="white", annot_kws={"fontsize": 11}, cbar_kws={"label": "Median within-readout z-score"}, ax=ax)
ax.set_xlabel("Target area"); ax.set_ylabel("Source area")
fig.tight_layout(); fig.savefig(ROOT / "figures/pdf/fig03_area_response_blocks.pdf"); plt.close(fig)
