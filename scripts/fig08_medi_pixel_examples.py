"""Figure 2a diagnostic: latent-space MEDI versus pixel-space optimization."""
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paper_figure_style import apply
A=ROOT/'data/assets/fig02_examples'; P=ROOT/'data/assets/figS06_pixel_examples'; OUT=ROOT/'figures/pdf/fig08_medi_pixel_examples.pdf'; AREAS=['V1','LM','RL','AL']; COLORS={'V1':'#2F5D8A','LM':'#C9792B','RL':'#A34F52','AL':'#4A8F67'}; FR=list(range(8,23,2))
def read(p):
 x=np.asarray(imageio.mimread(p)); x=x[...,:3].mean(-1) if x.ndim==4 else x; return x/(255 if x.max()>1 else 1)
def main():
 apply()
 # Register the actual Times New Roman face before selecting it.  Matplotlib's
 # font cache may not expose the Windows family name until addfont is called.
 times_font = Path('/mnt/c/Windows/Fonts/times.ttf')
 try:
  font_available = times_font.exists()
 except OSError:
  font_available = False
 if font_available:
  font_manager.fontManager.addfont(times_font)
  plt.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman']})
 plt.rcParams.update({'pdf.fonttype':42,'font.size':8,'savefig.bbox':'tight','savefig.pad_inches':.03})
 fig=plt.figure(figsize=(7.1,4.6)); gs=fig.add_gridspec(11,1,left=.17,right=.99,top=.88,bottom=.06,hspace=0,height_ratios=[1,1,.22,1,1,.22,1,1,.22,1,1]); axes=[]
 for j,a in enumerate(AREAS):
  for k,folder in enumerate([A,P]):
   x=read(folder/(f'{a}_medi.gif' if k==0 else f'{a}_pixel.gif')); strip=np.concatenate([x[i-1] for i in FR],axis=1); ax=fig.add_subplot(gs[3*j+k]); axes.append(ax); ax.imshow(strip,cmap='gray',vmin=0,vmax=1,aspect='auto',interpolation='nearest'); ax.axis('off'); ax.set_xlim(0,strip.shape[1]); ax.set_ylim(strip.shape[0],0)
   if j == 0 and k == 0:
    w=x.shape[2]
    for ii,frame in enumerate(FR): ax.text((ii+.5)*w, 1.08, str(frame), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=10)
    ax.text(-0.035, 1.08, 'frame', transform=ax.get_xaxis_transform(), ha='right', va='bottom', fontsize=10.5)
 fig.canvas.draw()
 for j,a in enumerate(AREAS):
  pa,pb=axes[2*j].get_position(),axes[2*j+1].get_position(); fig.text(.125,(pa.y0+pb.y1)/2,a,ha='right',va='center',color=COLORS[a],fontsize=11)
 p0,p1=axes[0].get_position(),axes[1].get_position(); fig.text(.125,(p0.y0+p0.y1)/2,'latent MEDI',ha='right',va='center',fontsize=10.5); fig.text(.125,(p1.y0+p1.y1)/2,'pixel MEDI',ha='right',va='center',fontsize=10.5); fig.savefig(OUT); plt.close(fig); print(OUT)
if __name__=='__main__': main()
