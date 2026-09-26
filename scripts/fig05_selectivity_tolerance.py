from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
import matplotlib.pyplot as plt
from table_pipeline import select_lowest_median_retention_parameter
from paper_figure_style import AREA_COLORS, apply, panel, save

apply()
plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "figures" / "tables"
OUT = ROOT / "figures" / "pdf"
d = pd.read_csv(TABLES / "selectivity_tolerance_by_parameter.csv")
order = ["contrast", "rotation", "translation", "temporal_reverse", "speed_up", "slow_down"]
labels = {"temporal_reverse": "temporal reverse", "speed_up": "speed up", "slow_down": "slow down"}
parameter_labels = {
    "contrast": "0.25x", "rotation": "90 deg", "translation": "48 px",
    "temporal_reverse": "-1", "speed_up": "2.0x", "slow_down": "0.5x",
}
areas = ["V1", "LM", "RL", "AL"]

chosen = select_lowest_median_retention_parameter(pd.read_csv(TABLES / "parameter_readouts.csv.gz"))
main = d.merge(chosen[["condition", "parameter"]], on=["condition", "parameter"])
main["condition"] = pd.Categorical(main.condition, order, ordered=True)
main = main.sort_values(["condition", "area"])
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1), sharey=True)
for i, (ax, cond) in enumerate(zip(axes.flat, order)):
    q = main[main.condition == cond].set_index("area").reindex(areas)
    ax.plot(range(len(areas)), q.rho, marker="o", color="#2F5D8A")
    ax.axhline(0, color="0.3", lw=.8)
    param = chosen.loc[chosen.condition == cond, "parameter"].iloc[0]
    ax.set_title(f"{labels.get(cond,cond)} ({parameter_labels[cond]})")
    ax.set_xticks(range(len(areas)), areas)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_ylim(-0.9, 0.08)
    ax.grid(axis="y", color="0.88", linewidth=.6)
    panel(ax, chr(97 + i))
save(fig, OUT / "fig05_selectivity_tolerance")
print(chosen.sort_values("condition").to_string(index=False))
