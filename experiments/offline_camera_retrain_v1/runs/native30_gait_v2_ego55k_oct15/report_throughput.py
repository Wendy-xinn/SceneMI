"""Estimate conditional wall time from actual full-pool Adam updates."""
import argparse
import datetime
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--minimum-samples',type=int,default=30)
    args=parser.parse_args()
    path=ROOT/'training/trace.jsonl'
    rows=[json.loads(line) for line in path.read_bytes().split(b'\n')[:-1] if line]
    # Latest occurrence if a future restart replays unsaved updates.
    unique={r['step']:r for r in rows};ordered=[unique[k] for k in sorted(unique)]
    measured=[r for r in ordered if r['step']>=3][-200:]
    if len(measured)<args.minimum_samples:
        raise SystemExit(f'Need{args.minimum_samples} measured full-pool updates; have{len(measured)}')
    def distribution(key):
        values=[r[key] for r in measured]
        return dict(mean=float(np.mean(values)),median=float(np.median(values)),p10=float(np.percentile(values,10)),p90=float(np.percentile(values,90)))
    fields={key:distribution(key) for key in ('preparation_s','compute_s','update_s')}
    compact=[]
    for r in measured:
        row={k:r[k] for k in ('step','loss','gradient_norm','lr','scene','timesteps','preparation_s','compute_s','update_s','support_cache_hits','support_cache_misses')}
        row['identities']=[{k:v[k] for k in ('group','mode','sequence_id','start','yaw_augmentation_rad')} for v in r['identities']]
        compact.append(row)
    (ROOT/'STARTUP_THROUGHPUT_TRACE.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in compact))
    latest=ordered[-1]['step'];mean=fields['update_s']['mean']
    # Conservative scenario acknowledges that contact-accepted solver work can
    # increase as the model learns; this multiplier is an assumption, not data.
    conservative=max(mean*1.5,fields['update_s']['p90'])
    old=json.loads((ROOT.parent/'native30_physics_schedule_probe_oct15/RESULTS.json').read_text())
    ddim=np.mean([old['aggregate']['all'][r]['current']['DDIM_seconds_mean'] for r in ('clean','noisy')])
    first_validation=json.loads((ROOT/'training/foot_monitor.jsonl').read_text().splitlines()[0])
    val_seconds=max(12*float(ddim)+10,2*first_validation['validation_seconds'])
    checkpoint_seconds=5.
    elapsed_nonupdates=max(0.,ordered[-1]['elapsed_s']-sum(r['update_s'] for r in ordered))
    now=datetime.datetime.now().astimezone()
    projections={}
    for target in (1000,10000,55000):
        remaining=max(0,target-latest);validations=target//1000-latest//1000
        saves=target//100-latest//100
        low=remaining*mean+max(0,validations)*val_seconds+max(0,saves)*checkpoint_seconds
        high=remaining*conservative+max(0,validations)*val_seconds*1.5+max(0,saves)*checkpoint_seconds*2
        projections[str(target)]=dict(remaining_steps=remaining,hours_from_now=[low/3600,high/3600],
            estimated_time_from_now=[(now+datetime.timedelta(seconds=s)).isoformat() for s in (low,high)],
            conditional_on_passing_reviews=target>1000,automatic_continuation=False)
    result=dict(status='measured_full_pool_updates',observed_at=now.isoformat(),latest_step=latest,measured_updates=len(measured),
        batch_size=2,scope='Actual full-pool stratified TRAIN sampling, input build, forward/loss/backward, gradient clip and Adam. Not the reused small-pool benchmark.',
        per_update_seconds=fields,compact_measured_trace='STARTUP_THROUGHPUT_TRACE.jsonl',observed_scene_off_fraction=float(np.mean([not r['scene'] for r in measured])),
        support_cache_misses=ordered[-1]['support_cache_misses'],
        observed_elapsed_non_update_seconds=elapsed_nonupdates,
        assumptions=dict(high_update_seconds=conservative,high_rate_explanation='max(observedp90,1.5xearly mean); allowance for learned physical contact/variable scene cost, not a statistical confidence interval.',
            validation_seconds_per1000=val_seconds,validation_basis='max(12 calls at trained10k sceneON/physicsON mean plus10s,2xactual first6-call validation); later actual monitoring can update this.',
            checkpoint_seconds_per100=checkpoint_seconds),
        remaining_walltime_projection=projections,
        limitation='Early30-200 updates cannot establish final runtime or convergence.55k is a projection and cap; this invocation stops at1000.')
    (ROOT/'THROUGHPUT_AND_ETA.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
