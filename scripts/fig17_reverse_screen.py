"""Generate the temporal-reverse screening figure (Figure S3)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from paper_figure_style import AREA_COLORS, apply, panel, save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "pdf"
TABLES = ROOT / "figures" / "tables"
QUALITY = TABLES / "quality_sensitivity_summary.csv"
AREAS = ["V1", "LM", "RL", "AL"]
VIVID = {"V1": "#1f77b4", "LM": "#ff7f0e", "RL": "#d62728", "AL": "#2ca02c"}
ORDER = ["other", "near-static candidate", "response-invariant despite large change"]
DISPLAY = {
    "other": "other",
    "near-static candidate": "near-static",
    "response-invariant despite large change": "large change",
}

apply()

# S2: mechanisms are classified with within-area thresholds, matching the appendix text.
mechanisms = pd.read_csv(TABLES / "temporal_reverse_two_mechanisms.csv")
mechanisms["display"] = pd.Categorical(
    mechanisms["mechanism"], categories=ORDER, ordered=True
).map(DISPLAY)
fig, ax = plt.subplots(figsize=(7.2, 4.25))
sns.boxplot(
    data=mechanisms,
    x="display",
    y="mean_pixel_distance",
    hue="brain_area",
    order=[DISPLAY[x] for x in ORDER],
    hue_order=AREAS,
    palette=VIVID,
    showfliers=False,
    linewidth=0.8,
    ax=ax,
)
ax.set_xlabel("temporal-reverse screen category")
ax.set_ylabel("mean forward/reverse pixel-distance proxy")
ax.legend(title="Area", frameon=False, ncol=2)
save(fig, OUT / "fig17_reverse_screen")

 
