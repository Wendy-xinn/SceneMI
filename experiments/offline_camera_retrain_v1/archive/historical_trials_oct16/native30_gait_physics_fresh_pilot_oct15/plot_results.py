"""Plots for the completed bounded fresh pilot; no smoothing."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d

ROOT=Path(__file__).resolve().parent
NAMES=['current','physics_off','late_four','late_two']


def main():
    r=json.loads((ROOT/'RESULTS.json').read_text())
    assert r['all_sampling_noise_and_lr_draws_paired']
    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(5)
    for ax,key,title in ((axes[0,0],'body_local_foot_error_mm','TRAIN body-local foot error (mm)'),
            (axes[0,1],'foot_velocity_RMSE_m_s','TRAIN foot velocity vector RMSE (m/s)'),
            (axes[1,0],'articulation_log_ratio_error','TRAIN amplitude log-ratio error')):
        vals=[r['summary']['current']['initial/TRAIN_selected/clean'][key]]+[r['summary'][n]['final/TRAIN_selected/clean'][key] for n in NAMES]
        ax.bar(x,vals,color=['#999999','#1678bb','#aaaaaa','#e3891c','#498f7d'])
        ax.set_xticks(x);ax.set_xticklabels(['initial']+NAMES,rotation=10);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];x=np.arange(4)
    for offset,regime,color in ((-.18,'clean','#1678bb'),(.18,'noisy','#c14989')):
        ax.bar(x+offset,[r['summary'][n]['final/VAL_fixed/'+regime]['mpjpe_mm'] for n in NAMES],width=.35,label=regime,color=color)
    ax.set_title('Small fixed VAL MPJPE (mm)');ax.set_xticks(x);ax.set_xticklabels(NAMES,rotation=10);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fresh V2 / paired 128-update small-pool diagnostic\n6 TRAIN windows; matched Adam reset, initialization, batches, noise and LR; not a benchmark')
    fig.tight_layout();fig.savefig(ROOT/'FRESH_PILOT_RESULTS.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,9),squeeze=False)
    def local(motion,rest):
        m=torch.from_numpy(motion)[None];rest=torch.from_numpy(rest)[None]
        with torch.no_grad():
            j=forward_kinematics(m,rest)[0];rot=rotation_from_6d(m[0,:,3:9])
            return torch.einsum('tji,tkj->tki',rot,j-j[:,:1])[:,(10,11),2].numpy()
    for i,dataset in enumerate(('TRUMANS','EgoBody','RICH')):
        with np.load(ROOT/'current/initial_predictions.npz') as a:
            curves={'GT':local(a[f'TRAIN_{i}_truth'],a[f'TRAIN_{i}_rest']),
                    'initial':local(a[f'TRAIN_{i}_prediction'],a[f'TRAIN_{i}_rest'])}
        for name in NAMES:
            with np.load(ROOT/name/'final_predictions.npz') as a:
                curves[name]=local(a[f'TRAIN_{i}_prediction'],a[f'TRAIN_{i}_rest'])
        for foot in range(2):
            ax=axes[i,foot]
            for name,color,style in (('GT','#111111','-'),('initial','#aaaaaa',':'),
                    ('current','#1678bb','-'),('late_four','#e3891c','--'),('physics_off','#9261a6','-.')):
                v=curves[name][:,foot];ax.plot(np.arange(len(v))/30,v,color=color,linestyle=style,label=name,linewidth=1.2)
            ax.set_title(dataset+' / '+('left' if foot==0 else 'right')+' foot, body-frame Z')
            ax.set_xlabel('Time (s)');ax.set_ylabel('Position (m)');ax.grid(alpha=.2)
            if i==0 and foot==0:ax.legend(ncol=2)
    fig.suptitle('Actual DDIM20 gait on selected TRAIN windows, before/after 128 updates\nOriginal 30Hz outputs; no smoothing; small-pool learning does not establish VAL gait quality')
    fig.tight_layout();fig.savefig(ROOT/'TRAIN_GAIT_LEARNING.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
