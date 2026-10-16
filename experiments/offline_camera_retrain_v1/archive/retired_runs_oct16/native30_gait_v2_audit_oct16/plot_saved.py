"""Plot original step1000 outputs without smoothing or model updates."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'native30_gait_v2_ego55k_oct15'


def local(motion, rest):
    joints = forward_kinematics(motion, rest)
    r = rotation_from_6d(motion[...,3:9])
    return torch.einsum('btji,btkj->btki',r,joints-joints[:,:,:1])[:,:,(10,11)]


def main():
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'step1000_validation_plan.json').read_text())['plan']
    fig,axes = plt.subplots(3,4,figsize=(18,10),constrained_layout=True)
    with np.load(ROOT/'step1000_validation.npz') as saved:
        assert int(saved['step']) == 1000
        for row,(case,entry) in enumerate((i,r) for i,r in enumerate(plan) if r['panel']=='fixed'):
            rest=torch.from_numpy(saved[f'{case}_rest'])[None]
            truth=torch.from_numpy(saved[f'{case}_truth'])[None]
            pred=torch.from_numpy(saved[f'{case}_clean_prediction'])[None]
            a,b=local(pred,rest)[0].numpy(),local(truth,rest)[0].numpy()
            clock=np.arange(len(a))/30
            for foot in (0,1):
                ax=axes[row,foot]
                ax.plot(clock,b[:,foot,2],color='black',label='GT')
                ax.plot(clock,a[:,foot,2],color='#226eac',label='step1000')
                ax.set_title(f"{entry['dataset']} / {'left' if foot==0 else 'right'} foot local z")
                ax.set_ylabel('m');ax.set_xlabel('s');ax.legend(loc='best',fontsize=8)
            ax=axes[row,2]
            for foot,color in ((0,'#c55c31'),(1,'#3a8a4a')):
                error=a[:,foot]-b[:,foot]
                ax.plot(clock,np.linalg.norm(error,axis=-1),color=color,label=f"{'L' if foot==0 else 'R'} total")
                dynamic=error-error.mean(0,keepdims=True)
                ax.plot(clock,np.linalg.norm(dynamic,axis=-1),color=color,linestyle='--',label=f"{'L' if foot==0 else 'R'} demeaned")
            ax.set_title('Local foot error / mean bias removed');ax.set_xlabel('s');ax.set_ylabel('m');ax.legend(fontsize=7)
            pr=rotation_from_6d(pred[...,3:9]);tr=rotation_from_6d(truth[...,3:9])
            angle=torch.rad2deg(torch.acos(((pr*tr).sum((-1,-2))-1).div(2).clamp(-1,1)))[0].numpy()
            axes[row,3].plot(clock,angle,color='#824a9e')
            axes[row,3].set_title('Root orientation error');axes[row,3].set_ylabel('degrees');axes[row,3].set_xlabel('s')
    fig.suptitle('Fresh gait V2 / step1000 / three fixed official VAL windows / clean input\nOriginal 30Hz predictions; no smoothing; each body uses its own root frame',fontsize=13)
    fig.savefig(ROOT/'STEP1000_GAIT.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
