"""Complete-backbone equivalence and cached TRAIN throughput, zero updates."""
import argparse
import contextlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,make_model,window_options,training_plan,sample_item,write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import cached_support
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
import experiments.offline_camera_retrain_v1.native30_observation_ik as ik
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow

@contextlib.contextmanager
def reference_runtime():
    proximal=ik.position_proximal;nearest=Native30SurfaceWindow.query_nearest
    ik.position_proximal=lambda *a,**kw:proximal(*a,**kw,compact=False)
    Native30SurfaceWindow.query_nearest=lambda self,v,r:self.query_wide(v,r,neighbors=1)
    try:yield
    finally:ik.position_proximal=proximal;Native30SurfaceWindow.query_nearest=nearest


def run(config_path):
    torch.set_num_threads(4);config=read_config(config_path)
    out=HERE/'runs/native30_runtime_oct14/CACHED_FULL_BACKBONE.json'
    report=dict(status='running',optimizer_steps=0,rows=[],config=str(config_path),
        contract='same canonical geometry, noise/timestep/weights; reference full IK and radius enumeration; no precision/iteration changes')
    write_json(out,report)
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=192,
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    state=torch.load(HERE/config['warm_start']['checkpoint'],map_location='cpu',weights_only=False)
    old_config=state['config'];weights=json.loads((HERE/old_config['prepared_output']/'effective_config.json').read_text())['observation_coefficients']
    model=make_model(config).cuda();model.load_state_dict(state['model']);del state
    items=[]
    for group in ('trumans','rich'):
        mi=next(i for i,_ in enumerate(data.members[group]) if len(data.starts_for(group,i,'ego')))
        start=int(data.starts_for(group,mi,'ego')[len(data.starts_for(group,mi,'ego'))//2])
        sample,identity=data.sample(group,'ego',member=mi,start=start,yaw=.7)
        sample,scene,asset=builder.build(data,sample,identity,'ego',(15,),noise_rng=np.random.default_rng(718))
        sample=cached_support(data,builder,sample,identity);items.append((sample,scene,asset,'ego'))
    batch=collate_observations(items);known=model_inputs(config,batch);t=torch.full((2,),300,device='cuda',dtype=torch.long)
    signal=model.alphas[t];target=training_motion(config,batch,known)
    torch.manual_seed(710);noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn_like(target)
    for variant in ('reference','optimized'):
        model.zero_grad(set_to_none=True);torch.manual_seed(771);torch.cuda.manual_seed_all(771);model.train();model.core.scene_embedding.eval()
        manager=reference_runtime() if variant=='reference' else contextlib.nullcontext()
        with manager:
            torch.cuda.synchronize();began=time.monotonic()
            prediction,details=model(noisy,t,known,use_scene=True,return_details=True)
            loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,weights)
            loss.backward();torch.cuda.synchronize()
            row=dict(variant=variant,seconds=time.monotonic()-began,loss=float(loss),physical_accepted=float(details['physical_accepted_steps'].sum()))
            gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
            if variant=='reference':baseline=(prediction.detach().cpu().clone(),float(loss),gradients)
            else:
                row.update(output_max_error=float((prediction.detach().cpu()-baseline[0]).abs().max()),
                    loss_abs_error=abs(float(loss)-baseline[1]),
                    gradient_max_error=max(float((v-baseline[2][k]).abs().max()) for k,v in gradients.items()),
                    gradients_close=all(torch.allclose(v,baseline[2][k],atol=1e-6,rtol=1e-4) for k,v in gradients.items()))
                row['failed_gradient_names']=[k for k,v in gradients.items() if not torch.allclose(v,baseline[2][k],atol=1e-6,rtol=1e-4)]
                print('COMPARISON',json.dumps(row),flush=True)
                report['rows'].append(row);write_json(out,report)
                if not torch.allclose(prediction.detach().cpu(),baseline[0],atol=5e-6,rtol=1e-5) or not row['gradients_close']:
                    raise AssertionError('Complete model output/gradient equivalence failed')
            report['rows'].append(row);write_json(out,report);print(json.dumps(row),flush=True)
    del gradients,baseline,items,batch,known,prediction,details,loss,terms
    rng=np.random.default_rng(config['seed']+991);plan=training_plan(data,config)
    for repeat in range(6):
        model.zero_grad(set_to_none=True);began=time.monotonic()
        pairs=[sample_item(data,builder,config,plan,rng) for _ in range(2)]
        items,identities=zip(*pairs);batch=collate_observations(items);known=model_inputs(config,batch)
        t=torch.full((2,),300,device='cuda',dtype=torch.long);signal=model.alphas[t];target=training_motion(config,batch,known)
        noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn_like(target)
        torch.cuda.synchronize();prepared=time.monotonic()
        model.train();prediction,details=model(noisy,t,known,use_scene=True,return_details=True)
        loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,weights);loss.backward();torch.cuda.synchronize()
        row=dict(variant='random_cached',repeat=repeat,preparation_s=prepared-began,compute_s=time.monotonic()-prepared,total_s=time.monotonic()-began,
            finite=bool(torch.isfinite(loss)),identities=[dict(group=i['group'],mode=i['mode'],sequence=i['sequence_id'],start=i['start']) for i in identities])
        report['rows'].append(row);write_json(out,report);print(json.dumps(row),flush=True)
    report['status']='passed';write_json(out,report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);run(p.parse_args().config)
