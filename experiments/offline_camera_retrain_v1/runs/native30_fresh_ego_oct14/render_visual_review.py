"""Native mesh contact sheets and fixed-panel metric curves from real VAL."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def render(folder):
    audit = json.loads((folder / 'VISUALIZATION_AUDIT.json').read_text())
    cases = sorted(audit['cases'], key=lambda c: (c['panel'] != 'fixed', ('trumans','egobody','rich').index(c['dataset'])))
    for regime in ('clean','noisy'):
        fig = plt.figure(figsize=(16,10), facecolor='white')
        for pi, case in enumerate(cases):
            with np.load(folder / case['mesh_file']) as c:
                ax = fig.add_subplot(2,3,pi+1,projection='3d')
                t = 96
                points = c['static']
                if len(points):
                    ax.scatter(points[:,0], points[:,2], points[:,1], s=.5, c='#b0b7c2', alpha=.2)
                for name, color, alpha in (('truth','#2474d7',.35),(regime,'#bc187e',.62)):
                    vertices = c[name + '_vertices'][t]
                    ax.add_collection3d(Poly3DCollection(vertices[c['faces']][:,:,[0,2,1]], facecolors=color, edgecolors='none', alpha=alpha))
                center = c['truth_joints'][t,0]
                all_points = np.concatenate((c['truth_joints'][t], c[regime + '_joints'][t]))
                radius = max(1.1, float(np.abs(all_points-center).max()) + .12)
                ax.set_xlim(center[0]-radius, center[0]+radius)
                ax.set_ylim(center[2]-radius, center[2]+radius)
                ax.set_zlim(center[1]-radius, center[1]+radius)
                ax.set_box_aspect((1,1,1))
                ax.view_init(elev=16,azim=-65)
                ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)');ax.set_zlabel('Y (m)')
                m = case['metrics'][regime]
                ax.set_title(f'{case["dataset"]} / {case["panel"]}\nwindow MPJPE {m["mpjpe_mm"]["value"]:.1f} mm', fontsize=11)
        fig.suptitle(f'Actual official VAL / step {audit["step"]} / {regime} / frame 96 of 192\nNo GT alignment or floor correction; native meshes, neutral hands/face', fontsize=13)
        fig.legend(handles=[Patch(color='#2474d7',label='GT'),Patch(color='#bc187e',label='Prediction'),Patch(color='#b0b7c2',label='Observed static surfaces')],loc='lower center',ncol=3)
        fig.tight_layout(rect=(0,.04,1,.92))
        path = folder / f'MESH_{regime.upper()}_STEP{audit["step"]}.png'
        fig.savefig(path,dpi=120);plt.close(fig)
        print(path,flush=True)
    root = folder.parent
    vals = [json.loads(x) for x in (root/'training/validation.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for regime,color in (('clean','#246ed0'),('noisy','#bc187e')):
        steps=[v['step'] for v in vals]
        for ax,metric,label in ((axes[0],'mpjpe_mm','Fixed ego MPJPE (mm)'),(axes[1],'foot_speed_during_GT_stance_m_s','Foot speed during GT stance (m/s)')):
            y=[v['summary'][f'fixed/ego/{regime}/scene_on'][metric]['value'] for v in vals]
            ax.plot(steps,y,'o-',label=regime,color=color)
            ax.set(xlabel='Optimizer updates',ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same fixed official VAL panel and seeds; three ego windows only')
    fig.tight_layout();fig.savefig(folder/'METRIC_TRENDS.png',dpi=140);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    render(p.parse_args().folder)
