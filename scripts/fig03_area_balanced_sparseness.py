"""Figure 3b: area-balanced library sparseness distributions in established-reference order."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_figure_style import apply
apply()
readouts = pd.read_csv(ROOT / "figures/tables/parameter_readouts.csv.gz",
                       usecols=["session", "scan_idx", "readout_id", "brain_area", "selectivity"])
readouts = readouts.drop_duplicates(["session", "scan_idx", "readout_id"])
areas = ["V1", "LM", "RL", "AL"]
colors = {"V1": "#2F5D8A", "LM": "#C9792B", "AL": "#4A8F67", "RL": "#A34F52"}
samples = {}
quartiles = {}
for area in areas:
    x = readouts.loc[readouts.brain_area.eq(area), "selectivity"].to_numpy()
    samples[area] = x
    quartiles[area] = np.quantile(x, [.25, .5, .75])
    print(f"Figure 3b {area}: median={quartiles[area][1]:.6f}, "
          f"quartiles=[{quartiles[area][0]:.6f}, {quartiles[area][2]:.6f}]")
fig, ax = plt.subplots(figsize=(3.55, 3.15))
parts = ax.violinplot([samples[a] for a in areas], positions=np.arange(len(areas)),
                      widths=.78, showmeans=False, showmedians=False,
                      showextrema=False, points=200)
for body, area in zip(parts["bodies"], areas):
    body.set_facecolor(colors[area]); body.set_edgecolor(colors[area])
    body.set_alpha(.78); body.set_linewidth(.5)
# Dashed quartile guides: Q1, median, and Q3 for each area.
violin_edges = {}
for j, (body, area) in enumerate(zip(parts["bodies"], areas)):
    vertices = body.get_paths()[0].vertices
    center = float(j)
    left = vertices[vertices[:, 0] <= center]
    right = vertices[vertices[:, 0] >= center]
    # Interpolate the actual violin boundaries at each percentile so the
    # dashed guides meet the density outline rather than using fixed lengths.
    def boundary(points, q, side):
        order = np.argsort(points[:, 1])
        y = points[order, 1]; x = points[order, 0]
        yu, idx = np.unique(y, return_index=True)
        xu = x[idx]
        return float(np.interp(q, yu, xu))
    q1, q2, q3 = quartiles[area]
    violin_edges[area] = [(boundary(left, q, "left"), boundary(right, q, "right"))
                          for q in (q1, q2, q3)]
    for k, (q, lw, alpha) in enumerate([(q1, .8, .65), (q2, 1.2, .95), (q3, .8, .65)]):
        lo, hi = violin_edges[area][k]
        ax.plot([lo, hi], [q, q], ls="--", lw=lw,
                color="0.12", alpha=alpha, zorder=4)
# Adjacent-area pairwise tests.  We use two-sided Mann--Whitney U tests with
# Bonferroni correction across the three adjacent comparisons.
bracket_base = max(np.max(samples[a]) for a in areas) + .045
for j, (left, right) in enumerate(zip(areas[:-1], areas[1:])):
    xl, xr = samples[left], samples[right]
    p = mannwhitneyu(xl, xr, alternative="two-sided").pvalue
    p_adj = min(1.0, 3.0 * p)
    if p_adj < 1e-3:
        label = "***"
    elif p_adj < 1e-2:
        label = "**"
    elif p_adj < .05:
        label = "*"
    else:
        label = "n.s."
    y = bracket_base + j * .040
    ax.plot([j, j, j + 1, j + 1], [y - .004, y, y, y - .004],
            color="0.2", lw=.9, clip_on=False)
    ax.text(j + .5, y + .004, label, ha="center", va="bottom", fontsize=10.5)
ax.set_ylim(0, bracket_base + .16)
ax.set_xticks(np.arange(len(areas)), areas)
ax.set_xlabel("Target model area"); ax.set_ylabel("MEDI-set stimulus selectivity")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures/pdf/fig03_area_balanced_sparseness.pdf"); plt.close(fig)
