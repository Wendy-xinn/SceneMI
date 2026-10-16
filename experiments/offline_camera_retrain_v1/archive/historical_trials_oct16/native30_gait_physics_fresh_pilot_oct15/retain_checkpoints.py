"""Keep compact rejected trials; retain Adam/RNG for useful diagnostic controls."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json,sha256

ROOT=Path(__file__).resolve().parent


def digest(values):
    d=hashlib.sha256()
    for k,v in values.items():
        a=v.detach().cpu().contiguous().numpy()
        d.update(k.encode());d.update(str(a.dtype).encode());d.update(str(a.shape).encode());d.update(a.tobytes())
    return d.hexdigest()


def main():
    torch.set_num_threads(4)
    result=json.loads((ROOT/'RESULTS.json').read_text())
    assert result['status']=='completed_bounded_fresh_small_pool'
    report=dict(policy='One last.pt/arm; current and late_four keep Adam/RNG as useful diagnostic controls only. Finished physics_off and late_two failed TRAIN triage and have no planned resume; retain model/config/RNG and mark evaluation-only.',
        original_models_untouched=True,free_before=shutil.disk_usage(ROOT).free,cases=[])
    for name in ('current','physics_off','late_four','late_two'):
        path=ROOT/name/'last.pt';before_bytes=path.stat().st_size
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert state['step']==state['planned_steps']==128
        before=digest(state['model'])
        strip=name in ('physics_off','late_two') and not all(result['TRAIN_learnability_triage_gates'][name].values())
        if strip:
            assert 'optimizer' in state
            del state['optimizer']
            state.update(evaluation_only=True,optimizer_removed_reason='Finished rejected bounded trial; no planned resume; model/RNG/config retained')
            save_checkpoint(state,path)
        del state;gc.collect()
        after=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert digest(after['model'])==before
        assert all(k in after for k in ('effective_config','torch_rng','cuda_rng','numpy_rng','initialization'))
        assert ('optimizer' in after)==(not strip)
        report['cases'].append(dict(arm=name,optimizer_preserved=not strip,evaluation_only=strip,
            model_sha256=before,model_unchanged=True,bytes_before=before_bytes,bytes_after=path.stat().st_size,
            checkpoint_sha256=sha256(path)))
        del after;gc.collect()
    report['free_after']=shutil.disk_usage(ROOT).free
    report['bytes_reclaimed']=sum(r['bytes_before']-r['bytes_after'] for r in report['cases'])
    write_json(ROOT/'CHECKPOINT_RETENTION_AUDIT.json',report)


if __name__=='__main__':main()
