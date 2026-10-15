"""Standalone summary of the isolated solver-increment candidate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent/'increment_smoothing'


def main():
    r=json.loads((ROOT/'RESULTS.json').read_text())
    names=['physics_off','current','current_smooth5','late_four','late_four_smooth5']
    labels=['No explicit physics','Current','Current + increment5','Last 4','Last 4 + increment5']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));x=np.arange(len(names))
    for ax,key,title in ((axes[0],'mpjpe_mm','Full-body MPJPE (mm)'),
            (axes[1],'foot_velocity_RMSE_m_s','Foot velocity error (m/s)'),
            (axes[2],'foot_acceleration_RMSE_m_s2','Foot acceleration error (m/s²)')):
        for offset,regime,color in ((-.18,'clean','#1678bb'),(.18,'noisy','#c14989')):
            ax.bar(x+offset,[r['aggregate']['all'][regime][n]['metrics'][key] for n in names],width=.35,label=regime,color=color)
        ax.set_title(title);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=22,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].legend()
    fig.suptitle('Same existing 10k weights / 9 development VAL windows / DDIM20\n5-frame filter on solver increments BEFORE original energy/observation acceptance; generated motion is not post-smoothed')
    fig.tight_layout();fig.savefig(ROOT/'INCREMENT_SMOOTHING.png',dpi=150)


if __name__=='__main__':main()
