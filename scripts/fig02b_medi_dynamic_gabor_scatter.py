"""Figure 2b: paired MEDI/dynamic-Gabor response scatter."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paper_figure_style import apply
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures/pdf/fig02b_medi_dynamic_gabor_scatter.pdf'; AREAS=['V1','LM','RL','AL']; COLORS={'V1':'#2F5D8A','LM':'#C9792B','RL':'#A34F52','AL':'#4A8F67'}
def main():
 apply(); plt.rcParams.update({'savefig.bbox':'tight','savefig.pad_inches':.03,'axes.spines.top':False,'axes.spines.right':False})
 d=pd.read_csv(ROOT/'data/baseline_responses.csv'); d=d.rename(columns={'medi_response':'medi','dynamic_gabor_response':'gabor'}).dropna(subset=['medi','gabor']); d=d[(d.medi<=80)&(d.gabor<=80)]
 fig,ax=plt.subplots(figsize=(3.25,3.05));
 for a in AREAS:
  q=d[d.brain_area==a]; ax.scatter(q.medi,q.gabor,s=4,alpha=.22,color=COLORS[a],label=a,edgecolors='none')
 hi=80; ax.plot([0,hi],[0,hi],'--',color='.45',lw=.7); ax.set(xlim=(0,hi),ylim=(0,hi),xlabel='MEDI response',ylabel='dynamic Gabor response'); ax.grid(color='.9',lw=.4); ax.legend(frameon=False,fontsize=9,markerscale=4,ncol=2,loc='upper left'); fig.savefig(OUT); plt.close(fig); print(f'{OUT}; n={len(d)}')
if __name__=='__main__': main()
