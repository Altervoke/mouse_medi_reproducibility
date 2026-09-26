"""Nested readout-split audit: select doses on one half, estimate on the other."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from paper_figure_style import AREA_COLORS, apply, panel, save

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "figures" / "tables"
OUT = REPO / "figures" / "pdf" / "fig15_nested_audit.pdf"
AREAS = ["V1", "LM", "RL", "AL"]
CONDS = ["contrast", "rotation", "translation", "temporal_reverse", "speed_up", "slow_down"]
LABELS = {
    "contrast": "contrast (0.25x)", "rotation": "rotation (90 deg)",
    "translation": "translation (48 px)", "temporal_reverse": "temporal reverse (-1)",
    "speed_up": "speed up (2.0x)", "slow_down": "slow down (0.5x)",
}

apply()
r=pd.read_csv(ROOT/'nested_selectivity_tolerance_audit.csv')
fig,axes=plt.subplots(2,3,figsize=(7.2,4.6),sharey=True)
for i,(ax,cond) in enumerate(zip(axes.flat,CONDS)):
    q=r[r.condition==cond]
    for area in AREAS:
        color=AREA_COLORS[area]
        v=q[q.area==area].sort_values('selection_half'); ax.plot(v.selection_half,v.rho,color=color,lw=1); ax.errorbar(v.selection_half,v.rho,yerr=[v.rho-v.bootstrap_ci_low,v.bootstrap_ci_high-v.rho],fmt='o',color=color,ms=3,capsize=2,label=area if cond=='contrast' else None)
    ax.axhline(0,color='.35',lw=.7); ax.set_title(LABELS[cond]); ax.set_xticks([0,1],['split A','split B']); ax.set_xlabel('selection half'); ax.set_ylabel('held-out Spearman rho'); ax.grid(axis='y', color='.88', lw=.6); panel(ax,chr(97+i))
handles, legend_labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, legend_labels, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.52,1.0))
fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(OUT); plt.close(fig)
