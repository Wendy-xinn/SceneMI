"""Read-only audit of body caches, splits, domain exposure and tracked motion."""
import json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from smplx.lbs import blend_shapes,vertices2joints
HERE=Path(__file__).parent
OUT=HERE/'runs/training_contract_audit_oct07'
PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
def fk(p,t,rest):
    local=Rotation.from_rotvec(p.reshape(-1,3)).as_matrix().reshape(len(p),22,3,3)
    rotations=[local[:,0]];joints=[t+rest[0]]
    for j in range(1,22):
        q=PARENTS[j];rotations.append(rotations[q]@local[:,j]);joints.append(joints[q]+np.einsum('tij,j->ti',rotations[q],rest[j]-rest[q]))
    return np.stack(joints,1)
def stats(v):
    v=np.asarray(v);return dict(n=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95)),max=float(v.max())) if len(v) else {}
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    models={};body=[];errors=defaultdict(list);sources=defaultdict(set)
    for split in ['train','validation','test']:
        for row in map(json.loads,(HERE/'data/poses'/f'{split}.jsonl').read_text().splitlines()):
            folder=Path(row['folder']);rest=np.load(folder/'rest_joints.npy');key=(row['model'],row['gender'])
            if key not in models:models[key]=load_model(*key)
            model=models[key]
            with torch.no_grad():expected=vertices2joints(model.J_regressor,model.v_template[None]+blend_shapes(torch.tensor([row['betas']],dtype=torch.float32),model.shapedirs[:,:,:10]))[0,:22].numpy()
            p=np.load(folder/'pose_axis_angle_world.npy',mmap_mode='r');t=np.load(folder/'translation_world.npy',mmap_mode='r');truth=np.load(Path(row['prepared_folder'])/'joints_world.npy',mmap_mode='r')
            indices=sorted({0,2*(len(p)//4),2*((len(p)-1)//2)})
            predicted=fk(p[indices],t[indices],rest);e=np.linalg.norm(predicted-truth[indices],axis=-1)
            errors[(split,row['dataset'])].extend(e.ravel().tolist())
            body.append(dict(split=split,dataset=row['dataset'],sequence_id=row['sequence_id'],model=row['model'],gender=row['gender'],rest_max_error_m=float(np.linalg.norm(expected-rest,axis=-1).max()),fk_mean_m=float(e.mean()),fk_max_m=float(e.max()),sampled_frames=indices))
        print('body audit',split,flush=True)
    from experiments.offline_camera_retrain_v1.data import SOURCE,OfflineSceneMIData
    trumans_motion=[]
    for row in map(json.loads,(SOURCE/'sequences.jsonl').read_text().splitlines()):
        sources[(row['dataset'],row['split'])].add(row.get('recording_id',row['sequence_id']))
        if row['dataset']!='trumans' or not row.get('object_tracks'):continue
        tracks=np.load(row['object_tracks'],allow_pickle=True).item()
        for name,tr in tracks.items():
            pos=np.asarray(tr['location']);rot=np.asarray(tr['rotation']);m=Rotation.from_euler('xyz',rot).as_matrix()
            angles=np.linalg.norm(Rotation.from_matrix(np.einsum('ij,tjk->tik',m[0].T,m)).as_rotvec(),axis=-1)
            trumans_motion.append(dict(sequence_id=row['sequence_id'],split=row['split'],object=name,max_translation_from_first_m=float(np.linalg.norm(pos-pos[0],axis=-1).max()),max_rotation_from_first_deg=float(np.rad2deg(angles.max()))))
    summary={f'{s}/{d}':stats(v) for (s,d),v in errors.items()}
    rest_summary={f'{s}/{d}':stats([r['rest_max_error_m'] for r in body if r['split']==s and r['dataset']==d]) for s in ['train','validation','test'] for d in ['trumans','egobody']}
    sizes={};rich_splits={}
    for split in ['train','validation']:
        data=OfflineSceneMIData(split,skeleton_profile='canonical_smpl',rich_source='native20_faceout_oct07');domains={}
        for dataset,groups in [('trumans',['trumans']),('egobody',['camera_wearer','interactee'])]:
            rows=[m for g in groups for m,v in data.base.groups[g] if any(v.values())];domains[dataset]=dict(frames=sum(r['frames_20fps'] for r in rows),sequences=len(rows))
        domains['rich']=dict(frames=sum(int(np.load(data.rich_root/m['sequence_id']/'joints_scenemi_yup.npy',mmap_mode='r').shape[0]) for m,v in data.rich_members),sequences=len(data.rich_members))
        total=sum(v['frames'] for v in domains.values())
        for v in domains.values():v['frame_share']=v['frames']/total;v['hours']=v['frames']/72000
        sizes[split]=domains
        rich_splits[split]=dict(scenes=sorted({m['scene'] for m,v in data.rich_members}),subjects=sorted({m['sequence_id'].split('_')[1] for m,v in data.rich_members}))
    overlaps={d:sorted(sources[(d,'train')]&sources[(d,'validation')]) for d in ['trumans','egobody']}
    report=dict(body_cache_rest_model_max_error_m=rest_summary,sampled_per_joint_fk_error_m=summary,body_cases=body,tracked_objects=trumans_motion,dynamic_summary=dict(objects=len(trumans_motion),moving_over_5cm_or_5deg=sum(r['max_translation_from_first_m']>.05 or r['max_rotation_from_first_deg']>5 for r in trumans_motion)),native20_usable_sizes=sizes,source_recording_train_val_overlap=overlaps,rich_split=rich_splits,rich_scene_overlap=sorted(set(rich_splits['train']['scenes'])&set(rich_splits['validation']['scenes'])),rich_subject_overlap=sorted(set(rich_splits['train']['subjects'])&set(rich_splits['validation']['subjects'])),limits='Three source-aligned frames per person sequence, not every frame; object motion measured across each original sequence, not necessarily inside every training window; rest model uses manifest betas/gender. No data or checkpoint modifications.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['body_cases','tracked_objects']},indent=2),flush=True)
if __name__=='__main__':main()
