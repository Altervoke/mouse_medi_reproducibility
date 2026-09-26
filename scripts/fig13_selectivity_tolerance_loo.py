"""Figure S13: selected-stress correlations with self-excluded selectivity."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from paper_figure_style import apply, panel, save

apply()
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "figures" / "tables"
OUT = ROOT / "figures" / "pdf"
AREAS = ["V1", "LM", "RL", "AL"]
ORDER = ["contrast", "rotation", "translation", "temporal_reverse", "speed_up", "slow_down"]
LABELS = {"temporal_reverse": "temporal reverse", "speed_up": "speed up", "slow_down": "slow down"}
PARAMETER_LABELS = {"contrast": "0.25x", "rotation": "90 deg", "translation": "48 px",
                    "temporal_reverse": "-1", "speed_up": "2.0x", "slow_down": "0.5x"}

def main():
    result = pd.read_csv(TABLES / "selectivity_tolerance_loo_by_parameter.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1), sharey=True)
    for i, (ax, condition) in enumerate(zip(axes.flat, ORDER)):
        q = result[result.condition.eq(condition)].set_index("area").reindex(AREAS)
        ax.plot(range(len(AREAS)), q.rho, marker="o", color="#2F5D8A")
        ax.axhline(0, color="0.3", lw=.8)
        ax.set_title(f"{LABELS.get(condition, condition)} ({PARAMETER_LABELS[condition]})")
        ax.set_xticks(range(len(AREAS)), AREAS)
        ax.set_ylabel("Spearman $\\rho$")
        ax.set_ylim(-0.9, .08)
        ax.grid(axis="y", color="0.88", linewidth=.6)
        panel(ax, chr(97 + i))
    save(fig, OUT / "fig13_selectivity_tolerance_loo")

if __name__ == "__main__":
    main()
