"""Zero-update cost probe: old-style backbone on clean official30Hz inputs.

This is architecture/runtime feasibility, not an old/new quality comparison.
Runs only after the paired optimization trial ends, so does not compete with it.
"""
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,make_model,window_options,write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import cached_support
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective,base_losses,observation_losses
from experiments.offline_camera_retrain_v1.native30_gait_supervision import DEFAULT
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.contact_supervision import calibrated_contact_loss

ROOT=Path(__file__).resolve().parent


def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert protocol['status'] in ('completed','failed_or_bounded_stop')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.28)
    config=read_config(HERE/'native30_fresh_ego_config.json')
    parent=torch.load(HERE/'runs/native30_fresh_ego_oct14/training/last.pt',map_location='cpu',weights_only=False,mmap=True)
    old=torch.load(HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False,mmap=True)
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    bank={};ids={}
    for group in ('trumans','camera_wearer','rich'):
        member=next(i for i in range(len(data.members[group])) if len(data.starts_for(group,i,'ego')))
        start=int(data.starts_for(group,member,'ego')[0])
        sample,identity=data.sample(group,'ego',member=member,start=start,yaw=.6)
        sample,scene,asset=builder.build(data,sample,identity,'ego',(15,))
        sample=cached_support(data,builder,sample,identity)
        bank[group]=(sample,scene,asset,'ego');ids[group]={k:identity[k] for k in ('sequence_id','start','source_start_30fps')}
    snapshot=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout
    report=dict(optimizer_steps=0,quality_comparison=False,fps=30,frames=192,batch_size=2,precision='FP32',identities=ids,
        initial_gpu_load=snapshot,scope='Three predetermined officialTRAIN B2 pairs; repeated full forward/objective/backward. Excludes prep, clipping, Adam, checkpoint and validation. External GPU load may affect timing.',
        lightweight='Old55k architecture, absolute x0 and one UNet pass, existing occupancy/BPS/head track;30Hz native base plus gait restoration and observation/contact losses; no body-dependent query/IK/physical solver. Old55k weights loaded only for zero-update runtime probe, not official quality/training initialization.',cases=[])
    for variant in ('current','legacy_backbone'):
        model=make_model(config).cuda() if variant=='current' else OfflineSceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True).cuda()
        model.load_state_dict(parent['model'] if variant=='current' else old['model'],strict=True)
        for pair in (('trumans','camera_wearer'),('trumans','rich'),('camera_wearer','rich')):
            batch=collate_observations([bank[k] for k in pair]);known=model_inputs(config,batch)
            times=torch.full((2,),300,device='cuda',dtype=torch.long);signal=cosine_alphas().cuda()[times]
            target=training_motion(config,batch,known) if variant=='current' else batch['motion']
            generator=torch.Generator(device='cuda').manual_seed(6621)
            noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn(target.shape,device='cuda',generator=generator)
            samples=[]
            for repeat in range(3):
                model.train();model.zero_grad(set_to_none=True);torch.manual_seed(10047)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                if variant=='current':
                    pred,details=model(noisy,times,known,use_scene=True,return_details=True)
                    loss,_=objective(pred,batch['motion'],batch,signal,details,config,parent['effective_config']['observation_coefficients'])
                else:
                    pred=model(noisy,times,batch,control_mask=fixed_control_mask(2,192,'head','cuda'),use_scene=True)
                    base=base_losses(pred,batch['motion'],batch,signal,gait_config=DEFAULT)
                    observed=observation_losses(pred,batch['motion'],batch,signal,config['observation'])
                    loss=base['total']+sum(parent['effective_config']['observation_coefficients'][k]*v for k,v in observed.items())
                    loss=loss+.1*calibrated_contact_loss(model.contact_logits(pred,batch),batch,signal)
                assert torch.isfinite(loss)
                loss.backward();torch.cuda.synchronize()
                elapsed=time.monotonic()-began
                peak=torch.cuda.max_memory_allocated()/2**30
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                samples.append(dict(seconds=elapsed,peak_CUDA_GiB=peak))
                del pred,loss
                if variant=='current':del details
                else:del base,observed
            row=dict(variant=variant,pair=pair,warmup=samples[0],measured=samples[1:])
            report['cases'].append(row);print(json.dumps(row),flush=True)
            del batch,known
        del model;gc.collect();torch.cuda.empty_cache()
    report['status']='passed_execution_and_finite_gradients'
    write_json(ROOT/'LIGHTWEIGHT_COST_PROBE.json',report)


if __name__=='__main__':main()
