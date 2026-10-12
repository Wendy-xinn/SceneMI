"""Read-only inventory/split, native-body and sampled scene-memory audit."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.native30_export import inventory
from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS, SOURCE, PARENTS
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow

RUN = HERE/'runs/exo_occlusion_review_oct12'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_key(row):
    return row['recording'] if row['dataset']=='egobody' else row['sequence_id']


def scene_key(row):
    return row.get('scene_family',row.get('scene_id',row.get('scene')))


def main():
    torch.set_num_threads(1);before=shutil.disk_usage(HERE).free
    rows={split:inventory(split) for split in ('train','validation')}
    groups=('trumans','camera_wearer','interactee','rich');summary={}
    for split,items in rows.items():
        summary[split]={}
        for group in groups:
            lengths=[r['raw_last_exclusive']-r['raw_first'] for r in items if r['group']==group]
            summary[split][group]=dict(scoped_segments=len(lengths),raw_frames=sum(lengths),
                segments_shorter_than_192=sum(n<192 for n in lengths),
                candidate_starts_upper_bound=sum(max(0,n-191) for n in lengths))
    separation={}
    for dataset in ('egobody','trumans','rich'):
        a=[r for r in rows['train'] if r['dataset']==dataset]
        b=[r for r in rows['validation'] if r['dataset']==dataset]
        ak,bk={source_key(r) for r in a},{source_key(r) for r in b}
        common=ak&bk
        overlap=[]
        for ra in a:
            if source_key(ra) not in common:continue
            for rb in b:
                if source_key(ra)!=source_key(rb):continue
                n=max(0,min(ra['raw_last_exclusive'],rb['raw_last_exclusive'])-max(ra['raw_first'],rb['raw_first']))
                if n:overlap.append(dict(train=ra['sequence_id'],validation=rb['sequence_id'],frames=n))
        separation[dataset]=dict(train_sources=len(ak),validation_sources=len(bk),
            shared_sources=sorted(common),overlap_pairs=overlap,
            train_scene_families=len({scene_key(r) for r in a}),validation_scene_families=len({scene_key(r) for r in b}),
            shared_scene_families=sorted({scene_key(r) for r in a}&{scene_key(r) for r in b}))
        if common:raise ValueError('Source separation changed; review cross-split sharing')
    official_path=PROJECTS/'egobody/data_splits.csv'
    official={}
    for r in csv.DictReader(official_path.open()):
        for split in ('train','val','test'):
            recording=r[split].strip()
            if recording:
                if recording in official:raise ValueError('Official recording occurs twice')
                official[recording]=split
    provenance={}
    for split,items in rows.items():
        ego=[r for r in items if r['dataset']=='egobody'];recordings=sorted({r['recording'] for r in ego})
        provenance[split]=dict(recordings_by_official_split=dict(Counter(official.get(r,'unlisted') for r in recordings)),
            scoped_segments_by_official_split=dict(Counter(official.get(r['recording'],'unlisted') for r in ego)),
            official_test_recordings=[r for r in recordings if official.get(r)=='test'])
    cache=Native30SkinAnchorCache(device='cpu');bodies={}
    for kind in ('smpl','smplx'):
        asset=cache.get(dict(model=kind,gender='male',betas=[0.]*10,scale=1.))
        parent=asset['parents'][:22].numpy()
        if not np.array_equal(parent,PARENTS):raise ValueError('Shared body22 topology differs')
        bodies[kind]=dict(native_vertices=len(cache.models[(kind,'male')].v_template),
                          common_joints=22,common_parent_tree_matches=True,anchors=len(asset['vertex_ids']),
                          anchor_ids=asset['vertex_ids'].tolist(),unsupported_extra_rotations_fixed=True)
    # Distinct points mark prefix, window and suffix; only window static points survive.
    points=np.array([[[f,0,0],[f,0,1]] for f in range(6)],np.float32)
    owner=np.tile([0,1],(6,1));camera=np.zeros((6,3),np.float32)
    surface=Native30SurfaceWindow(points[2:5],owner[2:5],camera[2:5])
    np.testing.assert_array_equal(surface.points,points[2:5,0])
    for t,cloud in enumerate(surface.dynamic):np.testing.assert_array_equal(cloud['points'],points[t+2,1:])
    scope=dict(source_frames=6,sampled_interval=[2,5],static_memory_frame_ids=surface.points[:,0].astype(int).tolist(),
               prefix_excluded=True,suffix_excluded=True,within_window_future_static_allowed=True,
               dynamic_memory='current frame only',static_voxel_m=.025,global_token_reuse='one DDIM call, not across windows')
    dependencies=[Path(__file__),HERE/'native30_data.py',HERE/'native30_observations.py',HERE/'native30_geometry.py',
                  HERE/'native30_config.json',official_path,HERE/'data/trumans_temporal_training_v3/exclusions.json']
    dependencies += [SOURCE/f'{dataset}_sequences/{split}.jsonl' for dataset in ('egobody','trumans') for split in rows]
    identity=[{k:r[k] for k in ('split','group','sequence_id','raw_first','raw_last_exclusive')} for values in rows.values() for r in values]
    result=dict(protocol='native30-sampling-memory-audit-v1',date='2026-10-12',training_steps=0,
                config_modified=False,source_data_modified=False,window_frames=192,fps=30,
                train_start_stride_candidates=1,train_overlap_allowed=True,validation_loop_implemented=False,
                exact_valid_window_count_available=False,candidate_counts_apply_source_validity_masks=False,
                source_summary=summary,split_separation=separation,egobody_official_split_provenance=provenance,
                official_egobody_benchmark_training_ready=False,native_body_compatibility=bodies,scene_memory_demo=scope,
                input_sha256={str(p):digest(p) for p in dependencies},
                inventory_identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
                disk_free_bytes_before=before,disk_free_bytes_after=shutil.disk_usage(HERE).free)
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'WINDOW_AND_MEMORY_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(source_summary=summary,split_separation=separation,
                         egobody_official_split_provenance=provenance,scene_memory_demo=scope),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
