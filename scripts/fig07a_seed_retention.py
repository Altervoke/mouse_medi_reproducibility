from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
import matplotlib.pyplot as plt
from paper_figure_style import AREA_COLORS, apply, save

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures/inputs"
OUT = ROOT / "figures/pdf"
apply()

corr = pd.read_csv(SRC / "cross_seed_condition_correlations.csv")
labels = corr["condition"].str.replace("_", " ")

fig, ax = plt.subplots(figsize=(4.4, 3.2))
x = range(len(corr))
ax.errorbar(x, corr.spearman_rho, yerr=[corr.spearman_rho-corr.ci_low, corr.ci_high-corr.spearman_rho], fmt="o", color="#222222", capsize=3)
ax.axhline(0, color="#999999", lw=.8)
ax.set_xticks(list(x), labels, rotation=35, ha="right")
ax.set(xlabel="Transformation condition", ylabel="Cross-seed Spearman rho", ylim=(-.05, 1.05))
save(fig, OUT / "fig07a_seed_retention")
