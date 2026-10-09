"""Export released RICH vertex contact at native20 timestamps with missing masks.

Discrete labels use nearest source frame (ties to earlier); never interpolated
as continuous motion. Preserve BOTH native mesh topologies. No invented zeros
are considered labeled. This export does not enable a model contact head.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent.resolve();RAW=HERE.parents[2]/'RICH/extracted'
def read_smplx(path):
    colors=[]
    with path.open() as f:
        for line in f:
            if line.startswith('v '):
                values=line.split()
                if len(values)==4:colors.append(None)
                elif len(values)==7:colors.append([float(x) for x in values[-3:]])
                else:raise ValueError(f'{path}: unexpected OBJ vertex format')
    if len(colors)!=10475:raise ValueError(f'{path}: wrong SMPL-X topology')
    if all(c is None for c in colors):return None
    if any(c is None for c in colors):raise ValueError(f'{path}: partial vertex labels')
    colors=np.asarray(colors)
    if colors.shape!=(10475,3):raise ValueError(f'{path}: wrong SMPL-X topology {colors.shape}')
    return np.all(np.isclose(colors,[0,1,0],atol=1e-6),axis=1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'data/rich_contact_native20_v1');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);records=[]
    for split in ['train','val']:
        paths=sorted((HERE/f'data/scene_visibility_v2_oct05/rich_{split}_smpl_native20_faceout_oct07').glob('*/metadata.json'))
        if a.limit:paths=paths[:a.limit]
        for path in paths:
            meta=json.loads(path.read_text());name=meta['sequence_id'];subject=meta['subject_id'];q=np.load(path.parent/'source_frame_ids.npy')
            ids=np.floor(q+.5-1e-9).astype(int);out=a.output/split/name;out.mkdir(parents=True,exist_ok=True)
            smpl=np.zeros((len(q),6890),bool);smplx=np.zeros((len(q),10475),bool);valid=np.zeros((len(q),2),bool)
            for t,frame in enumerate(ids):
                hsc=RAW/f'{split}_hsc'/name/f'{frame:05d}'/f'{subject}.pkl'
                if hsc.is_file():
                    value=np.asarray(pickle.load(hsc.open('rb'),encoding='latin1')['contact'])
                    if value.shape!=(6890,) or not np.isfinite(value).all() or not np.isin(value,[0,1]).all():raise ValueError(f'{hsc}: invalid contact')
                    smpl[t]=value.astype(bool);valid[t,0]=True
                obj=hsc.with_suffix('.obj')
                if obj.is_file():
                    value=read_smplx(obj)
                    if value is not None:smplx[t]=value;valid[t,1]=True
            np.savez_compressed(out/'contacts.npz',smpl_packed=np.packbits(smpl,axis=1),smplx_packed=np.packbits(smplx,axis=1),valid=valid,source_frame_ids=q,label_frame_ids=ids)
            row=dict(sequence_id=name,split=split,frames=len(q),smpl_labeled=int(valid[:,0].sum()),smplx_labeled=int(valid[:,1].sum()),source=str(RAW/f'{split}_hsc'/name),temporal_protocol='nearest original 30Hz frame; ties earlier; <=0.5 source-frame offset; explicit missing mask; native topology preserved',model_contact_prediction_enabled=False)
            (out/'metadata.json').write_text(json.dumps(row,indent=2));records.append(row)
            (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));print('contact exported',split,name,row['smpl_labeled'],row['smplx_labeled'],flush=True)
if __name__=='__main__':main()
