"""Generate Figure 4: transformation-specific area-family retention profiles."""
from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
REPO=Path(__file__).resolve().parents[1]; REPO=REPO
sys.path.insert(0,str(REPO/"src"))
from paper_figure_style import apply
OUT=REPO/"figures"/"pdf"/"fig04_area_family_profiles.pdf"
STATS_OUT=REPO/"figures"/"tables"/"adjacent_area_statistics.csv"
AREAS=["V1","LM","RL","AL"]; CONDS=["contrast","rotation","translation","temporal_reverse","speed_up","slow_down"]
DISPLAY={"temporal_reverse":"temporal reverse","speed_up":"speed up","slow_down":"slow down"}
apply(); plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                              "axes.titlesize": 14, "xtick.labelsize": 12,
                              "ytick.labelsize": 12})
d=pd.read_csv(REPO/"figures"/"tables"/"parameter_readouts.csv.gz"); d.loc[d.condition.eq("temporal_reverse"),"parameter"]=-1.0; d=d[d.condition.isin(CONDS)]
keys=["session","scan_idx","readout_id","brain_area","condition"]
f=(d[~d.condition.isin(["speed_up","slow_down"])].groupby(keys,observed=True).retention.median().reset_index().groupby(["brain_area","condition"],observed=True).retention.agg(median="median",sd="std",n="size").reset_index())
# For speed conditions, aggregate the multiple factors within each readout
# first.  The uncertainty unit is therefore a readout, not the four factor
# values in the area-summary table (which would spuriously make n=4).
sr=pd.read_csv(REPO/"figures"/"inputs"/"variable_speed_readout_summary.csv")
sr["condition"]=np.where(sr.speed_factor>1,"speed_up","slow_down")
rk=["session","scan_idx","readout_id","brain_area","condition"]
sr=sr.groupby(rk,observed=True).retention.median().reset_index()
s=sr.groupby(["brain_area","condition"],observed=True).retention.agg(median="median",sd="std",n="size").reset_index()
f=pd.concat([f,s],ignore_index=True)
# Readout-level values used both for the plotted medians and all comparisons.
readout=pd.concat([
    d[~d.condition.isin(["speed_up","slow_down"])].groupby(keys,observed=True).retention.median().reset_index(),
    sr
],ignore_index=True)
rng=np.random.default_rng(20260926); pairs=list(zip(AREAS[:-1],AREAS[1:])); rows=[]
def bootstrap_difference(cond,left,right,n_boot=5000):
    x=readout.loc[(readout.condition==cond)&(readout.brain_area==left),"retention"].dropna().to_numpy(float)
    y=readout.loc[(readout.condition==cond)&(readout.brain_area==right),"retention"].dropna().to_numpy(float)
    obs=float(np.median(x)-np.median(y))
    # Batch the resampling to keep memory bounded for the largest area.
    b=np.empty(n_boot); batch=250
    for start in range(0,n_boot,batch):
        stop=min(start+batch,n_boot)
        bx=rng.choice(x,(stop-start,x.size),replace=True); by=rng.choice(y,(stop-start,y.size),replace=True)
        b[start:stop]=np.median(bx,axis=1)-np.median(by,axis=1)
    lo,hi=np.quantile(b,[.025,.975]); p=(2*min(np.sum(b<=0),np.sum(b>=0))+1)/(n_boot+1)
    # The uncertainty interval is bootstrap-based; the two-sided assessment
    # uses the readout-level Mann--Whitney test, avoiding a finite-resample
    # p-value floor for the large, highly separated cohorts.
    mw=float(mannwhitneyu(x,y,alternative="two-sided").pvalue)
    return obs,float(lo),float(hi),mw,x.size,y.size,float(min(1,p))
for cond in CONDS:
    for left,right in pairs:
        obs,lo,hi,p,nl,nr,bp=bootstrap_difference(cond,left,right)
        x=readout.loc[(readout.condition==cond)&(readout.brain_area==left),"retention"]
        y=readout.loc[(readout.condition==cond)&(readout.brain_area==right),"retention"]
        rows.append(dict(transformation=cond,contrast=f"{left}-{right}",left_area=left,right_area=right,
                         left_median=float(x.median()),right_median=float(y.median()),median_difference=obs,
                         ci_low=lo,ci_high=hi,p_value=p,bootstrap_p_value=bp,n_left=nl,n_right=nr,bootstrap_replicates=5000,seed=20260926))
stats=pd.DataFrame(rows); stats["p_value_holm"]=multipletests(stats.p_value,method="holm")[1]
stats["significant"]=stats.p_value_holm<.05
stats["direction"]=np.where(stats.median_difference>0,"left_greater",np.where(stats.median_difference<0,"right_greater","tie"))
stats.to_csv(STATS_OUT,index=False)
def sig_label(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
fig,axes=plt.subplots(2,3,figsize=(7.2,4.1),sharex=True,sharey=True); xpos=np.arange(4)
for i,(ax,cond) in enumerate(zip(axes.flat,CONDS)):
 q=f[f.condition==cond].set_index("brain_area").reindex(AREAS); center=q["median"].to_numpy(float); spread=q["sd"].to_numpy(float); ax.plot(xpos,center,color="#2F5D8A",marker="o",lw=1.2,ms=4.2); ax.fill_between(xpos,center-spread,center+spread,color="#2F5D8A",alpha=.18,linewidth=0); ax.set_title(DISPLAY.get(cond,cond)); ax.set_xticks(xpos,AREAS); ax.set_ylabel("Median retention"); ax.set_ylim(.25,1.25); ax.grid(axis="y",color="0.88",lw=.6); ax.text(-.24 if i % 3 else -.15,1.02,chr(97+i),transform=ax.transAxes,fontweight="bold",va="bottom",fontsize=14,clip_on=False)
 sub=stats[stats.transformation==cond].reset_index(drop=True)
 for j,row in sub.iterrows():
  y=1.06+0.06*j; ax.plot([j,j+1],[y,y],color="0.25",lw=.8,clip_on=False); ax.plot([j,j],[y-.014,y],color="0.25",lw=.8,clip_on=False); ax.plot([j+1,j+1],[y-.014,y],color="0.25",lw=.8,clip_on=False); ax.text(j+.5,y+.014,sig_label(row.p_value_holm),ha="center",va="bottom",fontsize=10,fontweight="bold",clip_on=False)
fig.tight_layout(); fig.savefig(OUT); plt.close(fig); print(f"wrote {OUT}")
