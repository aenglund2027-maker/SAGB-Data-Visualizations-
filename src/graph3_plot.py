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


def plot_graph3(df, output=None, show=True):
    order=['0–29%','30–49%','50–69%','70–100%']
    sub=df.set_index('form_bin').reindex(order)
    x=np.arange(len(order))
    fig,ax=plt.subplots(figsize=(11.2,6.5),facecolor='white')
    style_axis(ax)
    ax.plot(x,sub.expected_rate,color=EXPECTED,lw=2.3,marker='o',ms=7,
            markeredgecolor='white',markeredgewidth=1.2,zorder=4,label='Ranking + surface expectation')
    ax.plot(x,sub.upset_rate,color=OBSERVED,lw=2.8,marker='o',ms=9,
            markeredgecolor='white',markeredgewidth=1.4,zorder=5,label='Actual upset rate')
    # Place direct labels on the last two points to reduce reliance on visual color matching.
    for i in [0,3]:
        ax.annotate(f'{sub.upset_rate.iloc[i]:.1f}%',(x[i],sub.upset_rate.iloc[i]),
                    xytext=(0,14),textcoords='offset points',color=OBSERVED,
                    ha='center',fontsize=11,fontweight='bold')
    ax.set_xticks(x,order)
    ax.set_yticks([25,30,35,40,45],['25%','30%','35%','40%','45%'])
    ax.set_ylim(23,48)
    ax.set_xlabel('Underdog win rate over previous 10 ATP matches',fontsize=12,color=INK,labelpad=20)
    ax.set_ylabel('Underdog win rate',fontsize=12,color=INK,labelpad=16)
    ax.set_title('Does Recent Form Reveal Hidden Favorites?',loc='left',fontsize=22,
                 fontweight='bold',color=INK,pad=30)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.03),ncol=2,frameon=False,
              fontsize=10.5,columnspacing=2.4,labelcolor=INK)
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
    df = pd.read_csv(root / 'outputs' / 'graph3_recent_form_summary.csv')
    plot_graph3(df, root / 'outputs' / 'graph3_recent_form', show=False)
    print('Saved PNG and SVG to', root / 'outputs')
