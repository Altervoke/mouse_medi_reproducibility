"""Figure S11c: cross-seed stimulus pixel-similarity distribution."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_figure_style import apply, save
apply()
d = pd.read_csv(ROOT / "figures/inputs/cross_seed_stimulus_distances.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.hist(d.pixel_pearson, bins=12, color="#6E7F80", edgecolor="white")
median = d.pixel_pearson.median(); ax.axvline(median, color="#222222", lw=1.2)
ax.set(xlabel="Frame-aligned pixel correlation", ylabel="Readouts")
ax.text(median - .012, ax.get_ylim()[1] * .92, f"median = {median:.3f}", ha="right", va="top", fontsize=11)
save(fig, ROOT / "figures/pdf/fig07c_seed_stimulus_similarity")
