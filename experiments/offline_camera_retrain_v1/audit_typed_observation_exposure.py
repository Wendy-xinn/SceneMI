"""Replay only the independent observation RNG; no body/scene data or GT read."""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.typed_head_condition import prepare_observation

OUT=Path(__file__).parent/'runs/typed_head_track_oct10'

def main():
    torch.set_num_threads(4)
    for label in ('joint_noise','joint_reliable'):
        c=json.loads((OUT/label/'config.json').read_text());g=torch.Generator().manual_seed(c['seed']+19731)
        trace=[json.loads(line) for line in (OUT/label/'sample_trace.jsonl').read_text().splitlines()]
        counts=dict(examples=0,clean_complete=0,biased_or_drifting=0,missing_interval=0,position_only=0)
        for row in trace:
            b=c['batch_size'];t=row['length'];x=torch.zeros(b,t,22,9);x[...,3]=x[...,7]=1
            batch=dict(trajectory=x,motion=torch.zeros(b,t,201),joints=torch.zeros(b,t,22,3))
            out=prepare_observation(batch,c['observation_protocol'],generator=g)
            meta=out['observation_meta'][:,:,15];head=out['trajectory'][:,:,15]
            noisy=(head[...,:3].abs().amax((1,2))>1e-6)
            gap=(meta[...,:2].sum(-1)==0).any(1);pos_only=(meta[...,1]==0).all(1)
            clean=~(noisy|gap|pos_only)
            for key,value in [('examples',b),('clean_complete',int(clean.sum())),('biased_or_drifting',int(noisy.sum())),('missing_interval',int(gap.sum())),('position_only',int(pos_only.sum()))]:counts[key]+=value
        cp=torch.load(OUT/label/'last.pt',map_location='cpu',weights_only=False)
        assert torch.equal(g.get_state(),cp['observation_rng_state']);del cp
        counts.update(replayed_rng_matches_checkpoint=True,clean_complete_fraction=counts['clean_complete']/counts['examples'],scope='simulated batch head slots before original control mask; includes examples whose head was unobserved')
        (OUT/(label+'_exposure.json')).write_text(json.dumps(counts,indent=2));print(label,counts)

if __name__=='__main__':main()
