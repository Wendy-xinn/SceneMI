"""Read-only independent cache audit and dynamic-disappearance regression.

Known future dynamic poses remain valid per-time inputs for prerecorded or
explicitly forecast scenarios. This audit makes no forecasting claim.
"""
import hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import temporal_bps
from experiments.offline_camera_retrain_v1.verify_causal_scene20_features import check

OUT=HERE/'runs/replan_transition_oct10'


def synthetic():
    empty=np.empty((0,3),np.float32)
    positions=np.zeros((3,3),np.float32);rot=np.repeat(np.eye(3,dtype=np.float32)[None],3,0)
    anchors=np.array([[0,0,0]],np.float32)
    dynamic=[np.array([[.1,0,0]],np.float32),np.array([[.8,0,0]],np.float32),empty]
    bps,valid=temporal_bps([empty]*3,dynamic,positions,rot,anchors,return_valid=True)
    np.testing.assert_allclose(bps[:2,0,0],[.05,.4],atol=1e-7)
    assert not valid[2].any() and not bps[2].any(), 'Vanished dynamic surface left a ghost'
    static=[np.array([[0,0,.6]],np.float32),empty,empty]
    kept,kv=temporal_bps(static,dynamic,positions,rot,anchors,return_valid=True)
    np.testing.assert_allclose(kept[2,0],[0,0,.3],atol=1e-7)
    assert kv.all(), 'Previously known static surface was forgotten'
    changed=[dynamic[0],np.array([[0,0,.02]],np.float32),dynamic[2]]
    alt,_=temporal_bps(static,changed,positions,rot,anchors,return_valid=True)
    np.testing.assert_array_equal(kept[0],alt[0])
    np.testing.assert_array_equal(kept[2],alt[2])
    return dict(dynamic_moves_without_old_surface=True,disappeared_dynamic_becomes_unknown=True,static_persists=True,changing_future_dynamic_cannot_modify_past_or_leave_trail=True)


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);regression=synthetic();groups={};counts={};excluded=0
    for manifest in [HERE/'data/native20_temporal_scenes_v4/manifest.jsonl',HERE/'data/trumans_temporal_training_v3/manifest.jsonl']:
        for row in map(json.loads,manifest.read_text().splitlines()):
            if row.get('status')=='excluded':excluded+=1;continue
            folder=Path(row['path']);m=json.loads((folder/'metadata.json').read_text())
            groups.setdefault(m['memory_bundle'],[]).append(str(folder));counts[m['dataset']]=counts.get(m['dataset'],0)+1
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(check,j) for j in groups.items()]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%25==0:print('scene memories checked',len(results),len(groups),flush=True)
    sources=('scene_visibility_v2.py','cache_causal_scene20_features.py','verify_causal_scene20_features.py','causal_scene.py')
    report=dict(status='passed',synthetic=regression,groups=len(results),sequences=sum(r['sequences'] for r in results),datasets=counts,excluded_bundles=excluded,
                reference_query_frames=sum(r['queries'] for r in results),maximum_bps_error=max(r['max_bps_error'] for r in results),
                checks='ALL recording memories reconstructed from owner==0 raw observations, independent earliest timestamp/voxel lexsort; beginning/middle/end cached BPS per bundle use timestamp-gated static plus CURRENT owner1..99 only',
                limitation='Sampled BPS timestamps, not every cached frame; depends on correctness of rendered owner labels. Prerecorded future dynamic observations are known-future diagnostic conditions, not online forecasting.',
                sources_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in sources},elapsed_s=time.monotonic()-start,details=results)
    (OUT/'scene_memory_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)

if __name__=='__main__':main()
