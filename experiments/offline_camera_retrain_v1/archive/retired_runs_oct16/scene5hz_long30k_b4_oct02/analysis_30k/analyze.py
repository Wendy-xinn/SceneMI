"""Read-only model diagnostics; writes analysis artifacts, never updates weights."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData, collate, GROUPS
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, supervised_losses
from experiments.offline_camera_retrain_v1.evaluate_sampling import predicted_motion_metrics

OUT = Path(__file__).resolve().parent
RUN = OUT.parent


def existing_analysis():
    curve = []
    for path in sorted((RUN / 'evaluations').glob('*/metrics.json')):
        report = json.loads(path.read_text())
        groups = list(report['groups'].values())
        primary = report['primary_variant']
        def mean(key, variant=primary):
            return float(np.mean([g['mean_variants'][variant][key] for g in groups]))
        truth_speed = np.mean([g['mean_ground_truth_proxy_metrics']['mean_joint_speed_m_per_frame'] for g in groups])
        curve.append(dict(step=report['checkpoint_step'],mpjpe=mean('fk_mpjpe_cm'),
                          head=mean('head_error_cm'),relative_head=mean('head_path_relative_error_cm'),
                          speed_ratio=mean('mean_joint_speed_m_per_frame')/truth_speed,
                          slide=mean('truth_stance_foot_slide_cm_per_frame'),
                          floating_cm=mean('support_floating_m')*100,
                          penetration_cm=mean('support_penetration_m')*100,
                          coverage=mean('contact_coverage'),
                          head_no_scene=mean('head_error_cm','head_without_scene'),
                          head_no_control=mean('head_error_cm','without_control')))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    steps = [r['step']/1000 for r in curve]
    for ax, keys, title in zip(axes.flat,
        [('mpjpe','head'), ('head','head_no_scene','head_no_control'), ('speed_ratio',),
         ('slide',), ('floating_cm','penetration_cm'), ('coverage',)],
        ['DDIM position error (cm)', 'Head error / condition ablation (cm)',
         'Generated / GT joint speed', 'GT-stance foot speed (cm/frame)',
         'GT-height support proxies (cm)', 'Near-support foot fraction']):
        for key in keys:
            ax.plot(steps, [r[key] for r in curve],label=key)
        if keys==('speed_ratio',):ax.axhline(1,color='gray',linestyle=':')
        ax.set_title(title);ax.set_xlabel('Training steps (thousands)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed 12 validation clips, 128 frames, DDIM-20; single seed')
    fig.savefig(OUT / 'training_curves.png',dpi=160);plt.close(fig)

    text=(RUN/'evaluations/step_030000/gallery.html').read_text()
    gallery=json.loads(text.split('const data=',1)[1].split(';\nconst $=',1)[0])
    diagnostics=[]
    for c in gallery['cases']:
        gt=np.asarray(c['tracks']['GT']);pr=np.asarray(c['tracks']['head_full_scene'])
        feet=[10,11]
        def foot_relative(j):
            v=np.diff((j[:,feet]-j[:,None,0])[:,:,[0,2]],axis=0)
            return float(np.linalg.norm(v,axis=-1).mean()*100)
        vg=np.diff(gt[:,feet],axis=0);vp=np.diff(pr[:,feet],axis=0)
        stance=np.linalg.norm(vg,axis=-1)<.01
        root_aligned=(pr-pr[:,0:1])-(gt-gt[:,0:1])
        diagnostics.append(dict(group=c['group'],identity=c['identity'],
            gt_foot_relative_speed_cm_frame=foot_relative(gt),
            predicted_foot_relative_speed_cm_frame=foot_relative(pr),
            gt_stance_speed_cm_frame=float(np.linalg.norm(vg,axis=-1)[stance].mean()*100),
            predicted_stance_speed_cm_frame=float(np.linalg.norm(vp,axis=-1)[stance].mean()*100),
            root_aligned_mpjpe_cm=float(np.linalg.norm(root_aligned,axis=-1).mean()*100)))
    c=gallery['cases'][0];gt=np.asarray(c['tracks']['GT']);pr=np.asarray(c['tracks']['head_full_scene'])
    displacement=gt[-1,0,[0,2]]-gt[0,0,[0,2]]
    direction=displacement/np.linalg.norm(displacement)
    fig,axes=plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
    time=np.arange(len(gt))/20
    for j,name,color in [(0,'pelvis','black'),(10,'left foot','tab:blue'),(11,'right foot','tab:orange')]:
        for data,style,label in [(gt,'--','GT'),(pr,'-','pred')]:
            axes[0].plot(time,data[:,j][:,[0,2]]@direction,style,color=color,label=f'{name} {label}')
    for data,style,label in [(gt,'--','GT'),(pr,'-','pred')]:
        axes[1].plot(time,(data[:,10,[0,2]]-data[:,11,[0,2]])@direction,style,label=label)
        velocity=np.linalg.norm(np.diff(data[:,10],axis=0),axis=-1)*100
        axes[2].plot(time[1:],velocity,style,label=label)
    axes[0].set_ylabel('Along-path position (m)');axes[1].set_ylabel('Left-right foot separation (m)')
    axes[2].set_ylabel('Left foot speed (cm/frame)');axes[2].axhline(1,color='gray',linestyle=':')
    for ax in axes:ax.legend(ncol=3);ax.grid(alpha=.2);ax.set_xlabel('Time (s)')
    fig.suptitle(f'30k walking example: {c["identity"]["sequence_id"]}; dashed=GT')
    fig.savefig(OUT/'walking_diagnostic.png',dpi=160);plt.close(fig)
    (OUT/'existing_summary.json').write_text(json.dumps(dict(curve=curve,foot_diagnostics=diagnostics),indent=2))


@torch.inference_mode()
def fresh_probe():
    torch.set_num_threads(4)
    checkpoint=torch.load(RUN/'evaluations/step_030000/inference.pt',map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];model=OfflineSceneMI(cfg['latent_dim'],tuple(cfg['dim_mults'])).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    original=json.loads((RUN/'evaluations/step_030000/metrics.json').read_text())
    original_ids={(c['identity']['sequence_id'],c['identity']['source_start_30fps'])
                  for g in original['groups'].values() for c in g['cases']}
    rows=[]
    for split,lengths in [('validation',(64,128,192)),('test',(128,))]:
        ds=OfflineSceneMIData(split,seed=20261002)
        for length in lengths:
            for gi,group in enumerate(GROUPS):
                for i in range(4):
                    sample,identity=ds.sample(length,group)
                    batch={k:v.cuda() for k,v in collate([sample]).items()}
                    masks={'head':fixed_control_mask(1,length,'head','cuda')}
                    if split=='validation' and length==128:
                        masks['head_no_scene']=masks['head']
                        for label,joints in [('head_sparse_pelvis',[0]),('head_sparse_wrists',[20,21])]:
                            mask=masks['head'].clone();mask[:,::5,joints]=True;masks[label]=mask
                    truth=predicted_motion_metrics(batch['motion'],batch,joints=batch['joints']*2)
                    for mode,mask in masks.items():
                        generated=ddim_sample(model,batch,length,steps=20,seed=20261002+gi*100+i,
                                              control_mask=mask,use_scene=mode!='head_no_scene')
                        assert torch.isfinite(generated).all()
                        losses=supervised_losses(generated,batch['motion'],batch)
                        joints=forward_kinematics(generated,batch['rest'])
                        m=dict(fk_mpjpe_cm=float(losses['fk_mpjpe_m']*100),
                               head_error_cm=float(losses['head_error_m']*100),
                               truth_stance_foot_slide_cm_per_frame=float(losses['foot_sliding_m_per_frame']*100),
                               **predicted_motion_metrics(generated,batch,joints=joints))
                        rows.append(dict(split=split,length=length,group=group,index=i,identity=identity,
                            mode=mode,metrics=m,truth=truth,
                            overlaps_fixed_clip=(identity['sequence_id'],identity['source_start_30fps']) in original_ids))
                    print('done',split,length,group,i,flush=True)
    summaries=[]
    for split,length,mode in sorted({(r['split'],r['length'],r['mode']) for r in rows}):
        members=[r for r in rows if (r['split'],r['length'],r['mode'])==(split,length,mode)]
        summary=dict(split=split,length=length,mode=mode,n=len(members),
            metrics={k:float(np.mean([r['metrics'][k] for r in members])) for k in members[0]['metrics']},
            per_group={g:{k:float(np.mean([r['metrics'][k] for r in members if r['group']==g]))
                          for k in members[0]['metrics']} for g in GROUPS})
        summary['speed_ratio']=summary['metrics']['mean_joint_speed_m_per_frame']/np.mean(
            [r['truth']['mean_joint_speed_m_per_frame'] for r in members])
        summaries.append(summary)
    report=dict(checkpoint_step=checkpoint['step'],seed=20261002,ddim_steps=20,
                note='Small diagnostic probe, not a full validation or test benchmark. Test is head-only.',
                summaries=summaries,cases=rows)
    (OUT/'fresh_probe.json').write_text(json.dumps(report,indent=2))
    for summary in summaries:print(json.dumps(summary),flush=True)


if __name__=='__main__':
    existing_analysis()
    fresh_probe()
