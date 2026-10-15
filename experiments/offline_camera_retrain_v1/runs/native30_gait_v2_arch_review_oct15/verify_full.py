"""Fresh full model, V2 gait objective, three runtime variants; zero updates."""
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.train_native30 import (
    HERE,read_config,make_model,model_inputs,training_motion,collate_observations,
    objective,write_json,sha256,model_digest,
)
from experiments.offline_camera_retrain_v1.native30_query_backend import attach_query_backend

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    config=read_config(HERE/'native30_fresh_ego_gait_v2_config.json')
    prior=HERE/'runs/native30_speed_gait_review_oct15'
    cached=torch.load(prior/'timing_bank.pt',map_location='cuda',weights_only=False)
    assert cached['data_audit_sha256']==sha256(prior/'DATA_AUDIT.json')
    bank=cached['bank'];del cached
    parent=torch.load(HERE/'runs/native30_fresh_ego_oct14/training/last.pt',map_location='cpu',weights_only=False,mmap=True)
    weights=parent['effective_config']['observation_coefficients']
    expected_initial=parent['initialization']['initial_model_sha256'];del parent
    torch.manual_seed(config['seed']);model=make_model(config).cuda()
    assert model_digest(model)==expected_initial
    report=dict(status='running',optimizer_steps=0,initial_model_sha256=expected_initial,
        initialization='fresh original seed; no trained weights loaded',
        observation_coefficients='original calibration reused only for matched runtime verification; new phase must recalibrate',
        config=copy.deepcopy(config),cases=[],timing_scope='V2 objective held constant; B2 FP32 forward/loss/backward only',
        query_source_sha256=sha256(HERE/'native30_exact_query.py'))
    write_json(ROOT/'FULL_MODEL_VERIFICATION.json',report)
    for pair in (('trumans','camera_wearer'),('trumans','rich'),('camera_wearer','rich')):
        batch=collate_observations([bank[k] for k in pair]);known=model_inputs(config,batch)
        t=torch.full((2,),300,device='cuda',dtype=torch.long);signal=model.alphas[t]
        target=training_motion(config,batch,known);generator=torch.Generator(device='cuda').manual_seed(6621)
        noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn(target.shape,device='cuda',generator=generator)
        reference=None
        for name,backend,compact,contact_only in (('reference','reference',False,False),
                ('exact_gpu','cuda_exact_radius_v1',False,False),
                ('exact_gpu_contact_regions','cuda_exact_radius_v1',False,True),
                ('exact_gpu_compact_ik','cuda_exact_radius_v1',True,False)):
            attach_query_backend(model,{'runtime':{'query_backend':backend}});model.joint_anchor_compact=compact
            model.guidance_config['query_contact_regions_only']=contact_only
            samples=[]
            for repeat in range(3):
                model.train();model.zero_grad(set_to_none=True);torch.manual_seed(10047)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                pred,details=model(noisy,t,known,use_scene=True,return_details=True)
                loss,terms=objective(pred,batch['motion'],batch,signal,details,config,weights)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite V2 loss')
                loss.backward();torch.cuda.synchronize();elapsed=time.monotonic()-began
                samples.append(dict(seconds=elapsed,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30))
                if repeat==2:
                    gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
                    if any(not torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('Nonfinite V2 gradient')
                    snapshot=(pred.detach().cpu(),float(loss.detach()),gradients)
                    scalar_terms={k:float(v.detach()) for k,v in terms['base'].items()}
                    physical=details['physical_accepted_steps'].tolist()
                del pred,details,loss,terms
            if reference is None:reference=snapshot
            failed=[k for k in gradients if not torch.allclose(gradients[k],reference[2][k],atol=1e-6,rtol=1e-4)]
            row=dict(pair=pair,variant=name,warmup=samples[0],measured=samples[1:],loss=snapshot[1],
                prediction_max_abs_error=float((snapshot[0]-reference[0]).abs().max()),
                loss_abs_error=abs(snapshot[1]-reference[1]),
                gradient_max_abs_error=max(float((g-reference[2][k]).abs().max()) for k,g in gradients.items()),
                failed_gradients=failed,physical_accepted_steps=physical,base_terms=scalar_terms)
            report['cases'].append(row);write_json(ROOT/'FULL_MODEL_VERIFICATION.json',report)
            print(json.dumps({k:v for k,v in row.items() if k!='base_terms'}),flush=True)
            row['accepted_equivalent']=row['prediction_max_abs_error']<=5e-6 and not failed
            if not row['accepted_equivalent'] and name!='exact_gpu_compact_ik':raise AssertionError('Runtime variant failed equivalence')
            if name in ('exact_gpu','exact_gpu_contact_regions') and (row['prediction_max_abs_error'] or row['loss_abs_error']):raise AssertionError('Exact query changed output/loss')
            del snapshot,gradients
        del reference,batch,known
    report['status']='passed_selected_exact_gpu_optional_changes_not_selected'
    report['selected_runtime']=dict(query_backend='cuda_exact_radius_v1',joint_anchor_compact=False,query_contact_regions_only=False)
    report['contact_region_numeric_equivalence']=True
    report['contact_region_performance_selection']='Optional only: full-forward timing mixed across3 pairs; no consistent aggregate speedup versus exactGPU alone.'
    write_json(ROOT/'FULL_MODEL_VERIFICATION.json',report)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        path=ROOT/'FULL_MODEL_VERIFICATION.json'
        if path.exists():
            value=json.loads(path.read_text());value.update(status='failed',error=repr(exc));write_json(path,value)
        raise
