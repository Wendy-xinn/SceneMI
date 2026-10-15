"""Preserve source/control; retain rejected pilot model as evaluation-only."""
import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

ROOT=Path(__file__).resolve().parent


def model_sha(state):
    digest=hashlib.sha256()
    for name,value in state.items():
        assert not torch.is_floating_point(value) or torch.isfinite(value).all()
        array=value.contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    result=json.loads((ROOT/'RESULTS.json').read_text())
    assert result['status']=='not_passed_do_not_extend'
    before=shutil.disk_usage(ROOT).free
    p=ROOT/'gait_restore/last.pt';size=p.stat().st_size
    state=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    original=model_sha(state['model'])
    if 'optimizer' in state:
        assert all(torch.isfinite(v).all() for d in state['optimizer']['state'].values() for v in d.values() if isinstance(v,torch.Tensor) and torch.is_floating_point(v))
        state['optimizer_parameter_steps']={str(k):float(v['step']) for k,v in state['optimizer']['state'].items()}
        del state['optimizer']
        state.update(resumable=False,evaluation_only=True,optimizer_removal_reason='Completed100-update pilot did not meet preregistered incremental gait gate; no continuation scheduled. Original10000 and control10100 remain resumable.')
        save_checkpoint(state,p)
    del state;gc.collect()
    final=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    assert model_sha(final['model'])==original and 'optimizer' not in final
    for key in ('data_rng','condition_rng','torch_rng','cuda_rng','numpy_rng'):assert key in final
    del final;gc.collect()
    control=torch.load(ROOT/'control/last.pt',map_location='cpu',weights_only=False,mmap=True)
    assert control['stage_step']==100 and 'optimizer' in control
    control_sha=model_sha(control['model'])
    del control;gc.collect()
    report=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
        retained_resumable=['../native30_fresh_ego_oct14/training/last.pt','control/last.pt','../native_dynamic_scene20_contact_55k_oct07/last.pt'],
        candidate='gait_restore/last.pt',candidate_evaluation_only=True,model_sha256_before_and_after=original,
        control_model_sha256=control_sha,candidate_bytes_before=size,candidate_bytes_after=p.stat().st_size,
        retained=['model','config','counts','stage plan','parent provenance','sampler','condition/Torch/CUDA/NumPy RNG','optimizer step audit'],
        free_before_bytes=before,free_after_bytes=shutil.disk_usage(ROOT).free)
    (ROOT/'CHECKPOINT_RETENTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
