"""Verify final models/RNG and remove rejected trials' unused Adam tensors."""
import hashlib
import json
import shutil
from pathlib import Path

import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.items()):
        digest.update(name.encode());digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parent
    report=dict(free_disk_before=shutil.disk_usage(root).free,trials=[],
        untouched='official55k, previous3k, source datasets, scene and body assets')
    for name in ('single_exo_baseline','single_exo_scene90','mixed_absolute_scene','mixed_reference_scene'):
        path=root/name/'last.pt';before=path.stat().st_size
        state=torch.load(path,map_location='cpu',weights_only=False)
        if any(not torch.isfinite(value).all() for value in state['model'].values() if torch.is_floating_point(value)):
            raise FloatingPointError(name+' nonfinite model')
        if 'optimizer' in state and any(not torch.isfinite(value).all() for item in state['optimizer']['state'].values()
            for value in item.values() if torch.is_tensor(value) and torch.is_floating_point(value)):
            raise FloatingPointError(name+' nonfinite optimizer')
        if not all(key in state for key in ('torch_rng','cuda_rng','condition_rng','data_rng')):
            raise ValueError(name+' missing replay state')
        digest=model_digest(state['model']);stripped=name in ('single_exo_scene90','mixed_absolute_scene')
        if stripped:
            state.pop('optimizer',None);state['evaluation_only']=True
            state['optimizer_removed_reason']='finished absolute-x0 diagnostic; no resume planned; model and replay evidence retained'
            save_checkpoint(state,path)
            retained=torch.load(path,map_location='cpu',weights_only=False)
            if model_digest(retained['model'])!=digest:raise AssertionError('Cleanup changed retained model')
            del retained
        row=dict(trial=name,step=state['step'],counts=state['counts'],model_finite=True,
            optimizer_removed=stripped,resumable=not stripped,replay_state_preserved=True,
            bytes_before=before,bytes_after=path.stat().st_size,model_tensor_sha256=digest)
        report['trials'].append(row);print(json.dumps(row),flush=True)
        del state
    report['free_disk_after']=shutil.disk_usage(root).free
    write_json(root/'CHECKPOINT_AUDIT.json',report)


if __name__=='__main__':main()
