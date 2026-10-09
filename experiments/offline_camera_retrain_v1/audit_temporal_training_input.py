"""One forward/backward compatibility check, no optimizer or checkpoint writes."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI

data=OfflineSceneMIData('validation',skeleton_profile='trumans_male_v2')
members=[(m,v) for m,v in data.base.groups['trumans'] if v[128]]
sequence_index=next(i for i,(m,v) in enumerate(members) if m['sequence_id']=='2023-02-18@22-17-40')
start_index=next(i for i,(_,s) in enumerate(members[sequence_index][1][128]) if s==300)
folder=HERE/'data/trumans_temporal_training_v3/validation/2023-02-18@22-17-40'
sample,identity=data.sample(128,'trumans',sequence_index=sequence_index,start_index=start_index,scene_bundle=folder)
assert all(np.isfinite(x).all() for x in sample.values())
device='cuda' if torch.cuda.is_available() else 'cpu'
batch={k:v.to(device) for k,v in collate([sample]).items()}
torch.manual_seed(2026)
model=OfflineSceneMI().to(device).eval()
prediction=model(torch.randn_like(batch['motion']),torch.tensor([500],device=device),batch)
loss=(prediction-batch['motion']).square().mean();loss.backward()
gradients=[p.grad for p in model.parameters() if p.grad is not None]
assert torch.isfinite(prediction).all() and all(torch.isfinite(g).all() for g in gradients)
assert any((g!=0).any() for g in gradients)
report=dict(identity=identity,shapes={k:list(v.shape) for k,v in sample.items()},
            forward_shape=list(prediction.shape),loss=float(loss.detach()),finite_gradients=True,
            gradient_tensors=len(gradients),optimizer_steps=0,device=device)
(folder/'training_adapter_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
