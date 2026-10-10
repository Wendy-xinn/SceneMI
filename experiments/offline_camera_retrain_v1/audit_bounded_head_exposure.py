"""Replay the fixed quota and observation RNG against completed checkpoints."""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation

ROOT=Path(__file__).parent/'runs/bounded_head_adaptation_oct10'

def main():
    torch.set_num_threads(4);report={}
    for label in ('joint_all','joint_staged','rotation_only','mild_only'):
        config=json.loads((ROOT/label/'config.json').read_text())
        g=torch.Generator().manual_seed(config['seed']+19731)
        count=dict(examples=0,clean_complete=0,position_only=0,biased_or_drifting=0,maximum_position_error_m=0.)
        for row in map(json.loads,(ROOT/label/'sample_trace.jsonl').read_text().splitlines()):
            b=config['batch_size'];t=row['length'];x=torch.zeros(b,t,22,9);x[...,3]=x[...,7]=1
            batch=dict(trajectory=x,motion=torch.zeros(b,t,201),joints=torch.zeros(b,t,22,3))
            out=prepare_bounded_observation(batch,config['observation_protocol'],generator=g,step=row['step'],seed=config['seed'])
            offset=out['trajectory'][:,:,15,:3]*2
            noise=offset.abs().amax((1,2))>1e-8
            missing=(out['observation_meta'][:,:,15,1]==0).all(1)
            count['examples']+=b;count['clean_complete']+=int((~noise&~missing).sum());count['position_only']+=int(missing.sum());count['biased_or_drifting']+=int(noise.sum())
            count['maximum_position_error_m']=max(count['maximum_position_error_m'],float(offset.norm(dim=-1).max()))
        cp=torch.load(ROOT/label/'last.pt',map_location='cpu',weights_only=False)
        assert cp['step']==1000 and torch.equal(g.get_state(),cp['observation_rng_state']);del cp
        fraction=count['clean_complete']/count['examples'];assert fraction==(1. if label in ('joint_all','joint_staged') else .8)
        count.update(clean_complete_fraction=fraction,rng_matches_checkpoint=True,scope='before original joint-control mask; includes uncontrolled head slots')
        report[label]=count
    (ROOT/'exposure_audit.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
