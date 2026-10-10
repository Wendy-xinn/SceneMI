"""Released-label coverage and actual support coverage; never starts training."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.contact_supervision import aggregate_contacts,vertex_regions
from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache

OUT=HERE/'runs/contact_coverage_audit_oct11'
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);models={};report={}
    for split in ('train','val'):
        counts=np.zeros(22,dtype=np.int64);valid=np.zeros(22,dtype=np.int64);n=0;raw=0;excluded=0;sequences=0;time_errors=[];merged_counts=np.zeros(22,dtype=np.int64);distal_positive=Counter()
        root=HERE/'data/scene_visibility_v2_oct05'/f'rich_{split}_smpl_native20_faceout_oct07'
        for p in sorted((HERE/'data/rich_contact_native20_v1'/split).glob('*/contacts.npz')):
            info=json.loads((root/p.parent.name/'metadata.json').read_text());gender=info['gender']
            if gender not in models:models[gender]=load_model('smplx',gender)
            model=models[gender];regions=vertex_regions(model)
            with np.load(p) as z:
                ids=np.load(root/p.parent.name/'source_frame_ids.npy');assert np.array_equal(ids,z['source_frame_ids'])
                target,mask=aggregate_contacts(z['smplx_packed'],z['valid'][:,1],regions,10475)
                merged,merged_mask=aggregate_contacts(z['smplx_packed'],z['valid'][:,1],vertex_regions(model,merge_distal=True),10475);merged_counts+=(merged*merged_mask).sum(0).astype(np.int64)
                counts+=(target*mask).sum(0).astype(np.int64);valid+=mask.sum(0);n+=len(ids);sequences+=1
                labels=np.unpackbits(z['smplx_packed'],axis=1)[:,:10475].astype(bool);labels &= z['valid'][:,1,None]
                dominant=model.lbs_weights.argmax(-1).numpy();raw+=int(labels.sum());excluded+=int(labels[:,dominant>=22].sum())
                for j in np.unique(dominant[dominant>=22]):distal_positive[int(j)]+=int(labels[:,dominant==j].sum())
                time_errors.extend(abs(ids-z['label_frame_ids']).tolist())
        report[split]=dict(sequences=sequences,frames=n,valid_region_entries=valid.tolist(),positive_region_entries=counts.tolist(),merged_distal_positive_region_entries=merged_counts.tolist(),excluded_vertex_positives_by_joint=dict(distal_positive),positive_fraction=(counts/np.maximum(valid,1)).tolist(),all_labeled_vertex_positives=raw,positives_outside_public22=excluded,label_timestamp_max_error_source_frames=max(time_errors,default=0))
    trace=[json.loads(x) for x in (HERE/'runs/fresh55k_scene_physics_oct11/winding_scene/trace.jsonl').read_text().splitlines()];groups=Counter();floor=Counter();cache=FloorCache();collisions={};collision_issues=[]
    for row in trace:
        for identity in row['identities']:
            group=identity['group'];groups[group]+=1;floor[group]+=cache.get(identity) is not None
            key=(identity['memory_bundle'],identity['source_start_30fps']);path=identity['scene_bundle']
            if key in collisions and collisions[key]!=path:collision_issues.append(dict(key=key,first=collisions[key],second=path))
            collisions[key]=path
    report['last_trial']=dict(training_examples_by_group=dict(groups),confident_floor_examples_by_group=dict(floor),floor_coverage_by_group={g:floor[g]/n for g,n in groups.items()},floor_cache_scene_key_collisions=collision_issues,rich_classification_available_only_for_group='rich',geometry_supported_regions=[7,8,10,11],geometry_absent_regions=[i for i in range(22) if i not in (7,8,10,11)])
    (OUT/'coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
