"""Remove only optimizer tensors after evaluation; retain model and audit state."""
import os,gc
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

def compact_evaluated_checkpoint(path):
 path=Path(path);checkpoint=torch.load(path,map_location='cpu',weights_only=False)
 before=path.stat().st_size
 if 'optimizer' not in checkpoint:return dict(path=str(path),before_bytes=before,after_bytes=before,already_compact=True)
 del checkpoint['optimizer'];checkpoint['storage_role']='evaluation/warm-start; completed trial optimizer removed'
 temporary=path.with_name('verified_compact.pt');save_checkpoint(checkpoint,temporary)
 verified=torch.load(temporary,map_location='cpu',weights_only=False)
 assert checkpoint['model'].keys()==verified['model'].keys()
 assert checkpoint['config']==verified['config'] and checkpoint['step']==verified['step']
 assert all(torch.equal(t,verified['model'][name]) for name,t in checkpoint['model'].items())
 assert set(checkpoint)==set(verified)
 os.replace(temporary,path)
 directory=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(directory)
 finally:os.close(directory)
 del checkpoint,verified;gc.collect()
 return dict(path=str(path),before_bytes=before,after_bytes=path.stat().st_size,model_verified=True,optimizer_removed=True)
