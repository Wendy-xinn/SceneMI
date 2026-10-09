"""Count the exact prepared corpora used by the old 55k checkpoint."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData

HERE=Path(__file__).parent
OUT=HERE/'runs/report55k_fullval_oct07'
def summary(rows):
    frames=np.array([r['frames'] for r in rows]);seconds=frames/20
    scenes=defaultdict(list)
    for r in rows:scenes[r['scene']].append(r)
    return dict(sequences=len(rows),frames_20hz=int(frames.sum()),person_hours=float(seconds.sum()/3600),duration_s=dict(min=float(seconds.min()),median=float(np.median(seconds)),p95=float(np.percentile(seconds,95)),max=float(seconds.max())),scenes={k:dict(sequences=len(v),frames=sum(r['frames'] for r in v),person_hours=sum(r['frames'] for r in v)/72000) for k,v in sorted(scenes.items())})
def main():
    report={}
    for split in ['train','validation','test']:
        data=OfflineSceneMIData(split,skeleton_profile='canonical_smpl',rich_source='legacy5interp')
        groups={}
        for group,members in data.base.groups.items():
            groups[group]=[dict(sequence_id=m['sequence_id'],frames=m['frames_20fps'],scene=m['scene_family'],recording=m.get('recording'),first=m.get('source_first_frame'),last=m.get('source_last_frame_exclusive')) for m,v in members]
        if split!='test':
            groups['rich']=[dict(sequence_id=m['sequence_id'],frames=int(np.load(data.rich_root/m['sequence_id']/'joints_scenemi_yup.npy',mmap_mode='r').shape[0]),scene=m['scene']) for m,v in data.rich_members]
        report[split]={g:summary(rows) for g,rows in groups.items()}
        ego=groups['camera_wearer']+groups['interactee'];report[split]['egobody']=summary(ego)
        intervals=defaultdict(list)
        for r in ego:intervals[r['recording']].append((r['first'],r['last']))
        union=0
        for values in intervals.values():
            last_end=-1
            for first,end in sorted(values):
                union+=max(0,end-max(first,last_end));last_end=max(last_end,end)
        report[split]['egobody']['unique_recordings']=len(intervals)
        report[split]['egobody']['union_recording_hours']=union/30/3600
        report[split]['window_summary']=data.summary()
        if split=='test':report[split]['window_summary'].pop('rich',None)
        report[split]['prepared_manifest_corpus']={}
        for dataset in ['trumans','egobody']:
            manifest=Path('../diffusion-motion-inbetweening/experiments/offline_sequence_v1/data')/f'{dataset}_sequences'/f'{split}.jsonl'
            raw=[json.loads(line) for line in manifest.read_text().splitlines() if line]
            report[split]['prepared_manifest_corpus'][dataset]=summary([dict(frames=r['frames_20fps'],scene=r['scene_family']) for r in raw])
    # Native corrected motion sizes are separate, never used to report the old checkpoint.
    native={}
    for split in ['train','val']:
        rows=[]
        for p in (HERE/'data/scene_visibility_v2_oct05'/f'rich_{split}_smpl_native20_faceout_oct07').glob('*/metadata.json'):
            m=json.loads(p.read_text());rows.append(dict(sequence_id=m['sequence_id'],frames=m['frames_20fps'],scene=m['scene']))
        native[split]=summary(rows)
    report['native20_rich_reference_only']=native
    report['notes']='Prepared data used by the experiment, not complete published datasets. Frame totals count person tracks once, not overlapping windows. Person-hours = frames/20/3600. EgoBody recording union merges intervals across both roles at 30 Hz. No RICH test data was integrated. Window starts: audited starts for TRUMANS/EgoBody; every integer 20 Hz offset for RICH.'
    (OUT/'dataset_stats.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
