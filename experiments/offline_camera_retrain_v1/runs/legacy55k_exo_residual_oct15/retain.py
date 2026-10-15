"""Retain a resumable control; failed residual candidates remain evaluation-only."""
import hashlib
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json

ROOT=Path(__file__).resolve().parent


def digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__=='__main__':
    results=json.loads((ROOT/'RESULTS.json').read_text());audit=[]
    for arm in ('no_scene','scene'):
        path=ROOT/arm/'last.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        before=path.stat().st_size;model_hash=digest(state['model'])
        strip=arm=='scene' and results['status']=='not_passed_do_not_extend'
        if strip:
            state.pop('optimizer',None)
            state.update(evaluation_only=True,resume_planned=False,optimizer_removed_reason='Completed bounded candidate did not pass registered gates; no extension planned')
            save_checkpoint(state,path)
            check=torch.load(path,map_location='cpu',weights_only=False)
            assert digest(check['model'])==model_hash
        audit.append(dict(arm=arm,step=state['step'],model_sha256=model_hash,model_finite=True,
            optimizer_retained='optimizer' in state,evaluation_only=strip,bytes_before=before,bytes_after=path.stat().st_size))
    write_json(ROOT/'CHECKPOINT_RETENTION.json',dict(status='passed',models=audit,
        original55k_untouched=results['old55k_unchanged'],Linux_free_GiB=shutil.disk_usage(ROOT).free/2**30,
        E_free_GiB=shutil.disk_usage('/mnt/e').free/2**30))
