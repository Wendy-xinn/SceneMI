"""Paired recording-cluster summaries; retain all registered windows."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from retain import digest
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json,sha256,code_hashes
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d

PARENT=Path(__file__).resolve().parent
ROOT=PARENT/'reference_anchor'


def main():
    result=json.loads((ROOT/'RESULTS.json').read_text())
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    reports={arm:json.loads((ROOT/arm/'VALIDATION.json').read_text()) for arm in protocol['arms']}
    paired={}
    for arm in protocol['arms']:
        rows=[json.loads(line) for line in (ROOT/arm/'training_log.jsonl').read_text().splitlines()]
        assert len(rows)==1000
        assert all(np.isfinite(row[k]) for row in rows for k in ('loss','grad','seconds'))
        paired[arm]=[r['sample_indices'] for r in rows]
    assert all(x==next(iter(paired.values())) for x in paired.values())
    statistics={}
    for regime in ('clean','noisy'):
        rows={arm:[r for r in report['cases'] if r['regime']==regime] for arm,report in reports.items()}
        current=rows['prior_scene']
        clusters={}
        for i,r in enumerate(current):clusters.setdefault((r['group'] if r['group'] not in ('camera_wearer','interactee') else 'egobody',r['recording']),[]).append(i)
        statistics[regime]={}
        for tag,control in (('scene_benefit','prior_only'),('prior_benefit','scene_only')):
            delta=np.array([a['metrics']['occluded_mm']-b['metrics']['occluded_mm'] for a,b in zip(rows[control],current)])
            members=list(clusters.values());rng=np.random.default_rng(2026101556);samples=[]
            for _ in range(10000):
                ids=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])
                samples.append(float(delta[ids].mean()))
            statistics[regime][tag]=dict(mean_paired_gain_mm=float(delta.mean()),improved_cases=int((delta>0).sum()),
                total_cases=len(delta),recording_clusters=len(clusters),cluster_bootstrap95_mm=np.quantile(samples,[.025,.975]).tolist(),
                method='10000 paired recording-cluster resamples, window-weighted mean; positive means improvement; descriptive development panel only')
        weights=np.array([r['metrics']['occluded_count'] for r in current])
        statistics[regime]['pooled_occluded_mm']={arm:float(np.average([r['metrics']['occluded_mm'] for r in entries],weights=weights)) for arm,entries in rows.items()}
        statistics[regime]['by_group']={g:{arm:float(np.mean([r['metrics']['occluded_mm'] for r in entries if r['group']==g])) for arm,entries in rows.items()} for g in sorted({r['group'] for r in current})}
    write_json(ROOT/'PAIRED_ANALYSIS.json',statistics)
    saved=torch.load(ROOT/'predictions.pt',map_location='cpu',weights_only=False)
    identities=json.loads((ROOT/'IDENTITIES.json').read_text())['validation']
    motions=dict(initial=saved['initial'],truth=saved['truth'],**{k:saved['outputs'][k] for k in protocol['arms']})
    local={}
    for name,m in motions.items():
        j=forward_kinematics(m,saved['rest'])
        local[name]=torch.einsum('btji,btkj->btki',rotation_from_6d(m[...,3:9]),j-j[:,:,:1])[:,:,(10,11)].numpy()
    feet=[]
    for i,row in enumerate(identities):
        target=local['truth'][i];ta=np.sqrt(((target-target.mean(0))**2).sum(-1).mean(0))
        z=target[:,:,2];gtcorr=float(np.corrcoef(z.T)[0,1]) if np.min(z.std(0))>1e-6 else None
        values={}
        for name,x in local.items():
            if name=='truth':continue
            x=x[i];amp=np.sqrt(((x-x.mean(0))**2).sum(-1).mean(0));z=x[:,:,2]
            values[name]=dict(body_local_foot_RMS_ratio=(amp/np.maximum(ta,1e-8)).tolist(),
                left_right_Z_correlation=float(np.corrcoef(z.T)[0,1]) if np.min(z.std(0))>1e-6 else None)
        feet.append(dict(row,GT_body_local_foot_RMS_m=ta.tolist(),GT_left_right_Z_correlation=gtcorr,models=values))
    write_json(ROOT/'FOOT_ARTICULATION_AUDIT.json',dict(scope='All registered EXO windows, not a dedicated walking benchmark',cases=feet))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(15,12),constrained_layout=True)
    selected=[]
    for row_index,group in enumerate(('trumans','camera_wearer','interactee','rich')):
        ids=[i for i,r in enumerate(identities) if r['group']==group and r['regime']=='clean']
        i=ids[min(2,len(ids)-1)];selected.append(identities[i])
        for ax,name in zip(axes[row_index],('initial','prior_only','prior_scene')):
            for foot,style in ((0,'-'),(1,'--')):
                ax.plot(np.arange(192)/30,local['truth'][i,:,foot,2],style,color='#333333',alpha=.6,label='GT '+str(foot))
                ax.plot(np.arange(192)/30,local[name][i,:,foot,2],style,color='#b52b91',label='prediction '+str(foot))
            ax.set_title(group+' | '+name);ax.set_xlabel('seconds');ax.set_ylabel('body-local foot Z (m)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.savefig(ROOT/'FOOT_CURVES.png',dpi=140);plt.close(fig)
    write_json(ROOT/'FOOT_CURVE_CASES.json',dict(selection='Third preselected input-occlusion quantile per group, clean; not selected by outcome',cases=selected))
    retention=[]
    for arm in protocol['arms']:
        path=ROOT/arm/'last.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==1000
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        model_hash=digest(state['model']);before=path.stat().st_size
        # Useful minimal observation control remains resumable. A candidate
        # which failed the full registered protocol has no extension planned.
        strip=arm!='observation_only' and result['status']=='not_passed_do_not_extend'
        if strip:
            state.pop('optimizer',None);state.update(evaluation_only=True,resume_planned=False,
                optimizer_removed_reason='Completed bounded candidate; registered combined protocol failed; no continuation planned')
            save_checkpoint(state,path)
            assert digest(torch.load(path,map_location='cpu',weights_only=False)['model'])==model_hash
        retention.append(dict(arm=arm,step=1000,model_sha256=model_hash,optimizer_retained='optimizer' in state,
            evaluation_only=strip,bytes_before=before,bytes_after=path.stat().st_size))
    write_json(ROOT/'CHECKPOINT_RETENTION.json',dict(status='passed',models=retention,old55k_untouched=True))
    assert code_hashes()==protocol['code_sha256']
    assert sha256(protocol['legacy_checkpoint'])==protocol['legacy_checkpoint_sha256']
    write_json(ROOT/'FINAL_AUDIT.json',dict(status='passed',matched_all1000_batch_indices=True,loss_grad_finite=True,
        source_hashes_unchanged=True,old55k_unchanged=True,original_full_data_audit=str(PARENT/'DATA_AUDIT.json'),
        optimizer_updates=4000,total_sample_slots=32000,unique_TRAIN_windows=128,
        official_TEST_examples_used=False,legacy_prior_overlap_disclosed=True))
    print(json.dumps(statistics,indent=2))


if __name__=='__main__':main()
