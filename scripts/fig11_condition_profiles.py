"""Generate the parameter-resolved retention profiles in Appendix Figure S08."""
from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[1]; REPO=REPO
sys.path.insert(0,str(REPO/"src"))
from paper_figure_style import AREA_COLORS, apply, panel
OUT=REPO/"figures"/"pdf"/"fig11_condition_profiles.pdf"; AREAS=["V1","LM","RL","AL"]; CONDS=["contrast","rotation","translation","temporal_reverse","speed_up","slow_down"]
DISPLAY={"temporal_reverse":"temporal reverse","speed_up":"speed up","slow_down":"slow down"}; XL={"contrast":"contrast multiplier","rotation":"rotation (deg)","translation":"translation (px)","temporal_reverse":"temporal reverse","speed_up":"temporal speed multiplier","slow_down":"temporal speed multiplier"}
def ticks(ax,c,v):
 v=np.asarray(v,float); ax.set_xticks(v); ax.set_xticklabels([f"{x:g}x" for x in v] if c in {"contrast","speed_up","slow_down"} else (["-1"] if c=="temporal_reverse" else [f"{x:g}" for x in v]));
 if c in {"contrast","rotation","translation"}:
  ax.tick_params(axis="x",labelrotation=85 if c=="rotation" else 35,labelsize=9)
  for label in ax.get_xticklabels(): label.set_horizontalalignment("right")
apply(); d=pd.read_csv(REPO/"figures"/"tables"/"parameter_readouts.csv.gz"); d.loc[d.condition.eq("temporal_reverse"),"parameter"]=-1.0; d=d[d.condition.isin(CONDS)]
p=(d[~d.condition.isin(["speed_up","slow_down"])].groupby(["brain_area","condition","parameter"],observed=True).retention.agg(median="median",sd="std").reset_index())
s=pd.read_csv(REPO/"figures"/"inputs"/"variable_speed_area_summary.csv"); s["condition"]=np.where(s.speed_factor>1,"speed_up","slow_down"); s=s.rename(columns={"speed_factor":"parameter","median_retention":"median","sd_retention":"sd"})[["brain_area","condition","parameter","median","sd"]]; p=pd.concat([p,s],ignore_index=True)
fig,axes=plt.subplots(2,3,figsize=(7.2,4.85),sharey=True)
for i,(ax,c) in enumerate(zip(axes.flat,CONDS)):
 q=p[p.condition==c]
 for area in AREAS:
  z=q[q.brain_area==area].sort_values("parameter"); x=z.parameter.to_numpy(float); ax.plot(x,z["median"],marker="o",lw=1.2,ms=3.5,label=area,color=AREA_COLORS[area]); ax.fill_between(x,z["median"]-z["sd"],z["median"]+z["sd"],alpha=.10,color=AREA_COLORS[area])
 ax.set_title(DISPLAY.get(c,c)); ax.set_xlabel(XL[c]); ax.set_ylabel("Median response retention"); ticks(ax,c,sorted(q.parameter.unique())); ax.grid(axis="y",color="0.88",lw=.6); panel(ax,chr(97+i))
h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.52,1.0)); fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(OUT); plt.close(fig); print(f"wrote {OUT}")
