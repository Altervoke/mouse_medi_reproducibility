"""Figure 2a: paired MEDI/dynamic-Gabor frame strips."""
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paper_figure_style import apply

ROOT=Path(__file__).resolve().parents[1]; ASSET=ROOT/'data/assets/fig02_examples'; OUT=ROOT/'figures/pdf/fig02a_medi_dynamic_gabor_examples.pdf'
AREAS=['V1','LM','RL','AL']; EXAMPLE_IDS={'V1':'6_4_r4059','LM':'6_6_r4588','RL':'6_6_r3432','AL':'6_7_r2905'}; COLORS={'V1':'#2F5D8A','LM':'#C9792B','RL':'#A34F52','AL':'#4A8F67'}
FRAME_IDS=list(range(8,23,2))
def arr(p):
 im=Image.open(p); frames=[]
 for i in range(getattr(im,'n_frames',1)):
  im.seek(i); frames.append(np.asarray(im.convert('RGB')))
 x=np.asarray(frames); return x[...,:3].mean(-1).astype(float)/255
def main():
 apply()
 plt.rcParams.update({'savefig.bbox':'tight','savefig.pad_inches':.03})
 # A narrow inter-neuron gutter separates areas; the two rows within each
 # area remain flush so the MEDI/control pairing is visually explicit.
 fig=plt.figure(figsize=(7.1,4.35)); gs=fig.add_gridspec(11,1,left=.17,right=.99,top=.91,bottom=.06,hspace=0,
     height_ratios=[1,1,.22,1,1,.22,1,1,.22,1,1])
 axes=[]
 for j,a in enumerate(AREAS):
  m=arr(ASSET/f'{a}_medi.gif'); g=arr(ASSET/f'{a}_dynamic_gabor.gif')[:,30:,-256:]
  for k,x in enumerate([m,g]):
   strip=np.concatenate([x[i-1] for i in FRAME_IDS],axis=1); row=2*j+k+j//1; ax=fig.add_subplot(gs[row]); axes.append(ax); ax.imshow(strip,cmap='gray',vmin=0,vmax=1,aspect='auto',interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, strip.shape[1]); ax.set_ylim(strip.shape[0], 0)
   for s in ['top','right','bottom']: ax.spines[s].set_visible(False)
   ax.spines['left'].set_color(COLORS[a]); ax.spines['left'].set_linewidth(1.4)
   if j==0 and k==0:
    width=x.shape[2]
    ax.set_xticks([(i+.5)*width for i in range(len(FRAME_IDS))],[str(i) for i in FRAME_IDS])
    ax.xaxis.tick_top(); ax.tick_params(axis='x',length=0,pad=1,labelsize=11)
    ax.text(-.015,1.02,'frame',transform=ax.transAxes,ha='right',va='bottom',fontsize=11)
  
 fig.canvas.draw()
 p0, p1 = axes[0].get_position(), axes[1].get_position()
 for j,a in enumerate(AREAS):
  pa=axes[2*j].get_position(); pb=axes[2*j+1].get_position()
  fig.text(.125,(pa.y0+pb.y1)/2,a,ha='right',va='center',color=COLORS[a],fontsize=14)
 fig.text(.125,(p0.y0+p0.y1)/2,'MEDI',ha='right',va='center',fontsize=13)
 fig.text(.125,(p1.y0+p1.y1)/2,'dynamic Gabor filter',ha='right',va='center',fontsize=13)
 fig.savefig(OUT); plt.close(fig); print(OUT)
if __name__=='__main__': main()
