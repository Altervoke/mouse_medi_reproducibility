"""Render all layer-by-condition estimates as a supplementary figure."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from paper_figure_style import HEATMAP_CMAP, apply, panel
ROOT=Path(__file__).resolve().parents[1]; d=pd.read_csv(ROOT/'figures'/'tables'/'area_layer_transformation_summary.csv'); d['layer']=d.layer.replace({'L23':'L2/3'})
areas,layers=['V1','LM','RL','AL'],['L2/3','L4','L5']; conds=['contrast','rotation','translation','temporal_reverse','speed_up','slow_down']; labels=['contrast','rotation','translation','temporal reverse','speed up','slow down']
s=d.groupby(['brain_area','layer','condition'],as_index=False).median(numeric_only=True); apply(); fig,axes=plt.subplots(3,2,figsize=(6.6,8.0))
for i,(ax,c,label) in enumerate(zip(axes.flat,conds,labels)):
 q=s[s.condition==c]; v=q.pivot(index='brain_area',columns='layer',values='median_retention').reindex(index=areas,columns=layers); n=q.pivot(index='brain_area',columns='layer',values='n').reindex(index=areas,columns=layers); a=v.copy().astype(object)
 for x in areas:
  for y in layers: a.loc[x,y]=f'{v.loc[x,y]:.2f}\n(n={int(n.loc[x,y]):,})'
 sns.heatmap(v,annot=a,fmt='',cmap=HEATMAP_CMAP,vmin=0,vmax=1.05,linewidths=.5,linecolor='white',cbar=False,ax=ax,annot_kws={'fontsize':9.2}); ax.set_title(label); ax.set_xlabel('Layer'); ax.set_ylabel('Area'); panel(ax,chr(97+i))
fig.subplots_adjust(left=.10, right=.84, bottom=.07, top=.96, wspace=.38, hspace=.42)
cbar_ax = fig.add_axes([.88, .25, .022, .50])
fig.colorbar(axes[0,0].collections[0], cax=cbar_ax, label='Median retention')
fig.savefig(ROOT/'figures'/'pdf'/'fig16_layer_tolerance.pdf', bbox_inches='tight')
plt.close(fig)
