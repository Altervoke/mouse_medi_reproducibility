"""Appendix scatter audit for MEDI and five stimulus baselines."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paper_figure_style import apply

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "baseline_responses.csv"
OUT = ROOT / "figures" / "pdf" / "fig09_baseline_scatter.pdf"
AREAS = ["V1", "LM", "RL", "AL"]
COLORS = {"V1":"#2F5D8A", "LM":"#C9792B", "RL":"#A34F52", "AL":"#4A8F67"}
BASELINES = [("dynamic_gabor_response", "dynamic Gabor"), ("static_gabor_response", "static Gabor"),
             ("MESI", "MESI"), ("grating_response", "drifting grating"), ("Natural", "natural"),
             ("static_gabor_response", "static Gabor", "MESI")]

def main():
    apply()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "ps.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": .03})
    d = pd.read_csv(DATA)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.0), sharex=False, sharey=False)
    for ax, item in zip(axes.flat, BASELINES):
        col, label = item[:2]; xcol = "medi_response" if len(item)==2 else item[2]
        q = d.dropna(subset=[xcol, col]).copy()
        if q.empty:
            ax.text(.5, .5, "current-run export pending", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("MEDI response"); ax.set_ylabel(f"{label} response")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            continue
        hi = 80
        for area in AREAS:
            z = q[q.brain_area == area]
            ax.scatter(z[xcol], z[col], s=4, alpha=.22, color=COLORS[area], edgecolors="none", label=area)
        ax.plot([0, hi], [0, hi], "--", color="0.45", lw=.7)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_xlabel(f"{xcol if xcol != 'medi_response' else 'MEDI'} response"); ax.set_ylabel(f"{label} response")
        ax.grid(color="0.9", lw=.4); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panel = [name for name, *_ in BASELINES].index(col)
        if len(item)==3: panel=5
        ax.text(-.14, 1.18, chr(97 + panel), transform=ax.transAxes, fontweight="bold", va="top")
    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=4,
                   markerfacecolor=COLORS[area], markeredgewidth=0, alpha=.55,
                   label=area)
        for area in AREAS
    ]
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=4,
               loc="upper center", bbox_to_anchor=(.5, .995),
               handletextpad=.5, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, .92), w_pad=1.1, h_pad=1.0)
    fig.savefig(OUT); plt.close(fig)
    print(f"wrote {OUT}; rows={len(d)}")

if __name__ == "__main__": main()
