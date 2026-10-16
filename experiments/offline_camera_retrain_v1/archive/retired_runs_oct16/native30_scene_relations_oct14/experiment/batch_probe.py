"""Zero-update real-TRAIN batch capacity probe, separate from matched VAL."""
import datetime as dt
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import time
import traceback
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import make_model,write_json,code_hashes
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data,build_bank,augment
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations

ROOT=Path(__file__).resolve().parent
report=dict(status='building',optimizer_steps=0,validation_changed=False,
    started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),rows=[],
    scope='Two fixed TRAIN windows (EgoBody master exo, RICH ego), repeated/independently augmented for capacity only; no held-out motion assessment',
    timing='single cold synchronized model forward/loss/backward; excludes batch construction and data bank build; concurrent VAL may affect timing',
    includes='all scene modules and scheduled physics; same trainable parameter set and persistent stored Adam tensors as selected pilot',
    fixed_timestep=300,gpu_memory_fraction=.50,code_sha256=code_hashes())
write_json(ROOT/'BATCH_CAPACITY_PROBE.json',report)
try:
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.50)
    saved=torch.load(ROOT/'candidate/last.pt',map_location='cpu',weights_only=False)
    config=saved['config'];plan=json.loads((HERE/'runs/native30_overnight_ik_oct14/experiment/window_plan.json').read_text())['train']
    chosen=[next(v for v in plan if (v['group'],v['mode'])==pair) for pair in [('camera_wearer','exo'),('rich','ego')]]
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    bank,ids=build_bank(scoped_data(config,'train'),builder,chosen,config)
    report['identities']=ids
    model=make_model(config).cuda();model.load_state_dict(saved['model'])
    prefixes=tuple(saved['meta']['trainable_common_prefixes'])+tuple(saved['meta']['trainable_scene_prefixes'])
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(prefixes))
    # Stored Adam tensors are materialized to include their actual GPU bytes;
    # no optimizer is constructed or stepped and the checkpoint is untouched.
    adam=[v.to('cuda') for state in saved['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
    report['adam_tensor_bytes']=sum(v.numel()*v.element_size() for v in adam)
    weights=json.loads((HERE/config['prepared_output']/'effective_config.json').read_text())['observation_coefficients']
    del saved;gc.collect();report['status']='probing';write_json(ROOT/'BATCH_CAPACITY_PROBE.json',report)
    for size in (2,4,8):
        row=dict(batch_size=size)
        try:
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            items=[augment(bank[i%2],ids[i%2],builder,config,np.random.default_rng(1400+i),noisy=True,yaw=True) for i in range(size)]
            batch=collate_observations(items);known=model_inputs(config,batch)
            times=torch.full((size,),300,device='cuda',dtype=torch.long);alpha=model.alphas[times]
            target=training_motion(config,batch,known);generator=torch.Generator(device='cuda').manual_seed(1426)
            noise=torch.randn(target.shape,device='cuda',generator=generator)
            state=alpha[:,None,None].sqrt()*target+(1-alpha[:,None,None]).sqrt()*noise
            model.train();model.core.scene_embedding.eval();torch.cuda.synchronize();begin=time.monotonic()
            prediction,details=model(state,times,known,use_scene=True,return_details=True)
            loss,terms=objective(prediction,batch['motion'],batch,alpha,details,config,weights)
            loss.backward();torch.cuda.synchronize()
            elapsed=time.monotonic()-begin
            finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            row.update(status='passed' if finite else 'nonfinite',seconds=elapsed,seconds_per_sample=elapsed/size,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,loss=float(loss),
                accepted_physical_iterations=float(details['physical_accepted_steps'].sum()),optimizer_steps=0)
            del items,batch,known,times,alpha,target,noise,state,prediction,details,loss,terms
        except torch.cuda.OutOfMemoryError:
            row.update(status='cuda_oom',traceback=traceback.format_exc())
        report['rows'].append(row);write_json(ROOT/'BATCH_CAPACITY_PROBE.json',report);print('BATCH_PROBE',json.dumps(row),flush=True)
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        if row['status']!='passed':break
    report.update(status='completed',finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(ROOT/'BATCH_CAPACITY_PROBE.json',report)
except BaseException:
    report.update(status='failed',traceback=traceback.format_exc())
    write_json(ROOT/'BATCH_CAPACITY_PROBE.json',report)
    raise
