"""Observe launch and seal evidence only after a real fresh optimizer update."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json

ROOT=Path(__file__).resolve().parent

def main():
    launcher=json.loads((ROOT/'launcher.json').read_text());pid=launcher['pid']
    began=time.monotonic();deadline=began+1800
    while time.monotonic()<deadline:
        checkpoint=ROOT/'training/last.pt'
        if checkpoint.exists():break
        process=Path(f'/proc/{pid}/stat')
        if not process.exists() or process.read_text().split(') ',1)[1].startswith('Z '):
            raise RuntimeError('Training process exited before first checkpoint; inspect training.log')
        print('WAITING_FIRST_UPDATE',pid,round(time.monotonic()-began,1),flush=True)
        time.sleep(20)
    else:raise TimeoutError('No first checkpoint within 30 minutes')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    bench=json.loads((ROOT/'FRESH_BACKBONE_BENCHMARK.json').read_text())
    origin=state['initialization']
    assert origin['type']=='from_scratch' and origin['source_checkpoint'] is None
    assert origin['optimizer_state_entries']==0 and origin['all_model_parameters_trainable']
    assert origin['initial_model_sha256']==bench['initial_model_sha256']
    assert state['step']>=1 and state['planned_steps']==10000
    assert state['effective_config']['config']['training_mode']=='ego'
    assert state['effective_config']['config']['initialization']=='from_scratch'
    assert all('/ego' in k for k in state['counts'])
    digest=hashlib.sha256()
    for name,value in state['model'].items():
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    trained=digest.hexdigest();assert trained!=origin['initial_model_sha256']
    opt=state['optimizer']['state'];assert opt
    optimizer_steps=sorted({int(v['step']) for v in opt.values()})
    assert optimizer_steps==[state['step']]
    rows=[json.loads(l) for l in (ROOT/'training/trace.jsonl').read_text().splitlines()]
    row=next(x for x in rows if x['step']==state['step'])
    assert torch.isfinite(torch.tensor([row['loss'],row['gradient_norm']])).all()
    report=dict(status='verified_real_fresh_update',verified_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),pid=pid,checkpoint_step=state['step'],planned_steps=state['planned_steps'],initialization=origin,trained_model_sha256=trained,checkpoint_bytes=checkpoint.stat().st_size,optimizer_entries=len(opt),optimizer_steps=optimizer_steps,counts=state['counts'],first_update={k:row[k] for k in ('step','loss','gradient_norm','lr','preparation_s','compute_s','update_s','support_cache_hits','support_cache_misses')},disk_free_after={str(p):__import__('shutil').disk_usage(p).free for p in (ROOT,Path('/mnt/e'))},next_action='First fixed/rotating DDIM20 validation; training continues afterward. Exo is not automatically launched.')
    write_json(ROOT/'STARTED_FROM_SCRATCH.json',report)
    launcher.update(status='training_first_update_verified',checkpoint_step=state['step'],initial_model_sha256=origin['initial_model_sha256'])
    write_json(ROOT/'launcher.json',launcher)
    print(json.dumps({k:v for k,v in report.items() if k!='initialization'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
