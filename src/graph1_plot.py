# Reproducible figure from Jeff Sackmann ATP yearly match files.
# Requires the summary CSV produced by: python src/build_analysis.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

# Color palette can be customized in one place.
INK = '#253740'
MUTED = '#63747C'
GRID = '#E6EBED'
HARD = '#315F70'
CLAY = '#CA684A'
GRASS = '#84945C'
OBSERVED = '#B64A39'
EXPECTED = '#71838B'


def style_axis(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#BBC6C9')
    ax.tick_params(axis='both',length=0,pad=11,labelcolor=MUTED,labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color=GRID,linewidth=.9)


def plot_graph1(df, output=None, show=True):
    order=['1–9','10–24','25–49','50–99','100–199','200+']
    colors={'Hard':HARD, 'Clay':CLAY, 'Grass':GRASS}
    fig,ax=plt.subplots(figsize=(11.2,6.5),facecolor='white')
    for surf in ['Hard','Clay','Grass']:
        sub=df[df.surface==surf].set_index('rank_gap_bin').reindex(order)
        x=np.arange(len(order))
        ax.plot(x,sub.upset_rate,color=colors[surf],lw=2.5,marker='o',ms=7,
                markeredgecolor='white',markeredgewidth=1.2,label=surf,zorder=3)
    style_axis(ax)
    ax.set_xticks(np.arange(len(order)),order)
    ax.set_ylim(12,52)
    ax.set_yticks([20,30,40,50],['20%','30%','40%','50%'])
    ax.set_xlabel('ATP ranking gap (places)',fontsize=12,color=INK,labelpad=20)
    ax.set_ylabel('Upset rate',fontsize=12,color=INK,labelpad=16)
    ax.set_title('How Ranking Gaps Shape ATP Upsets',loc='left',fontsize=22,
                 fontweight='bold',color=INK,pad=30)
    ax.legend(ncol=3,loc='upper right',bbox_to_anchor=(1,1.03),frameon=False,
              fontsize=11,handlelength=2.1,columnspacing=2.5,labelcolor=INK)
    fig.subplots_adjust(left=.125,right=.94,bottom=.18,top=.78)
    if output:
        output=Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output.with_suffix('.png'),dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.22)
        fig.savefig(output.with_suffix('.svg'),facecolor='white',bbox_inches='tight',pad_inches=.22)
    if show:plt.show()
    else:plt.close(fig)



if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / 'outputs' / 'graph1_surface_ranking_gap_summary.csv')
    plot_graph1(df, root / 'outputs' / 'graph1_ranking_gap_surface', show=False)
    print('Saved PNG and SVG to', root / 'outputs')
