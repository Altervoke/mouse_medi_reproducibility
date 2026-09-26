"""Figure S11b: cross-seed area-profile agreement."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_figure_style import AREA_COLORS, apply, save
apply()
d = pd.read_csv(ROOT / "figures/inputs/cross_seed_area_profiles.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.2))
for area, q in d.groupby("brain_area"):
    ax.scatter(q.median_seed42, q.median_seed1042, s=28, label=area, color=AREA_COLORS[area], edgecolor="white", linewidth=.4)
lo = min(d.median_seed42.min(), d.median_seed1042.min()); hi = max(d.median_seed42.max(), d.median_seed1042.max()); pad = .04 * (hi - lo)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#777777", lw=.8)
ax.set(xlabel="Seed 42 area median", ylabel="Seed 1042 area median"); ax.legend(frameon=False, ncol=2, loc="lower right")
save(fig, ROOT / "figures/pdf/fig07b_seed_area_profiles")
