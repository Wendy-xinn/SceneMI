"""Same full model, batches, dropout and noise; zero optimizer updates."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile_full import *
import experiments.offline_camera_retrain_v1.native30_query_runtime as runtime
from exact_gpu_query import query_surfaces_cuda as candidate


def main():
    fresh='--fresh' in sys.argv
    deterministic='--deterministic' in sys.argv
    diagnostic='--audit-nondeterminism' in sys.argv
    output=ROOT/('NONDETERMINISTIC_REPEAT_AUDIT.json' if diagnostic else (('DETERMINISTIC_' if deterministic else '')+('FRESH_EXACT_QUERY_COMPARISON.json' if fresh else 'CULLED_EXACT_QUERY_COMPARISON.json')))
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.28)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    config=read_config(HERE/'native30_fresh_ego_config.json')
    source=HERE/'runs/native30_fresh_ego_oct14/training/last.pt'
    parent=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    bank={};sizes={}
    bank_path=ROOT/'timing_bank.pt'
    audit_sha=sha256(ROOT/'DATA_AUDIT.json')
    if bank_path.exists():
        saved=torch.load(bank_path,map_location='cuda',weights_only=False)
        assert saved['data_audit_sha256']==audit_sha and saved['config']==config
        bank=saved['bank'];sizes=saved['sizes'];del saved
    else:
        data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=192,
            memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
            exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
        builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
        for group in ('trumans','camera_wearer','rich'):
            member=next(i for i in range(len(data.members[group])) if len(data.starts_for(group,i,'ego')))
            start=int(data.starts_for(group,member,'ego')[0])
            sample,identity=data.sample(group,'ego',member=member,start=start,yaw=.6)
            sample,scene,asset=builder.build(data,sample,identity,'ego',(15,))
            sample=cached_support(data,builder,sample,identity);bank[group]=(sample,scene,asset,'ego')
            base=getattr(scene,'base',scene)
            sizes[group]=dict(static=len(base.static['points']),dynamic_total=sum(len(c['points']) for c in base.dynamic),dynamic_max=max(len(c['points']) for c in base.dynamic))
            print(group,sizes[group],flush=True)
        torch.save(dict(bank=bank,sizes=sizes,config=config,data_audit_sha256=audit_sha),bank_path)
    report=dict(status='running',optimizer_steps=0,checkpoint_updated=False,scene_sizes=sizes,
        parent_sha256=sha256(source),candidate_sha256=sha256(ROOT/'exact_gpu_query.py'),cases=[],
        deterministic_algorithms=deterministic,
        model_initialization='fresh original seed, no parent weights loaded' if fresh else 'parent 10000 weights',
        contract='All original radii, neighbor counts, sorted-ID FPS tie rules and float precision retained. Only spatial search implementation changes.',
        timing_scope='Full forward/objective/backward B2 FP32 30Hz/L192; excludes preparation, Adam, save and validation')
    write_json(output,report)
    torch.manual_seed(config['seed']);model=make_model(config).cuda()
    if fresh:
        from experiments.offline_camera_retrain_v1.train_native30 import model_digest
        initial_sha=model_digest(model)
        if initial_sha!=parent['initialization']['initial_model_sha256']:
            raise AssertionError('Fresh initialization differs from original sealed random weights')
        report['initial_model_sha256']=initial_sha
    else:model.load_state_dict(parent['model'],strict=True)
    coefficients=parent['effective_config']['observation_coefficients'];del parent
    original=runtime.query_surfaces_cuda
    for pair in (('trumans','camera_wearer'),('trumans','rich'),('camera_wearer','rich')):
        batch=collate_observations([bank[k] for k in pair]);known=model_inputs(config,batch)
        t=torch.full((2,),300,device='cuda',dtype=torch.long);signal=model.alphas[t]
        generator=torch.Generator(device='cuda').manual_seed(6621);target=training_motion(config,batch,known)
        noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn(target.shape,device='cuda',generator=generator)
        captured=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            # YawSurface transforms vector fields in-place after this return.
            captured.append((args,kwargs,{k:v.copy() for k,v in result.items()}))
            return result
        def compute():
            model.train();model.zero_grad(set_to_none=True);torch.manual_seed(10047)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
            prediction,details=model(noisy,t,known,use_scene=True,return_details=True)
            loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,coefficients)
            loss.backward();torch.cuda.synchronize()
            result=dict(seconds=time.monotonic()-began,loss=float(loss.detach()),
                peak_CUDA_GiB=torch.cuda.max_memory_allocated()/2**30,
                accepted_physical_steps=details['physical_accepted_steps'].tolist())
            return result,prediction.detach().cpu(),{k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
        runtime.query_surfaces_cuda=capture
        baseline0,pred,grad=compute();runtime.query_surfaces_cuda=original
        baseline1,_,repeat_grad=compute()
        baseline_jitter=max(float((grad[k]-repeat_grad[k]).abs().max()) for k in grad)
        del repeat_grad
        baseline=[baseline1,compute()[0]]
        for args,kwargs,expected in captured:
            actual=candidate(*args,**kwargs)
            bad=[k for k in expected if not np.array_equal(expected[k],actual[k])]
            if bad:raise AssertionError(('Changed real query fields',pair,bad))
        runtime.query_surfaces_cuda=candidate
        candidate0,_,_=compute();changed,newpred,newgrad=compute();more=compute()[0]
        runtime.query_surfaces_cuda=original
        error=float((pred-newpred).abs().max())
        failed=[k for k in grad if k not in newgrad or not torch.allclose(grad[k],newgrad[k],atol=1e-6,rtol=1e-4)]
        if set(grad)!=set(newgrad):failed.append('gradient parameter set')
        row=dict(pair=pair,original_warmup=baseline0,original_measured=baseline,
            candidate_warmup=candidate0,candidate_measured=[changed,more],
            exact_real_query_cases=len(captured),all_query_fields_bitwise_equal=True,
            prediction_max_abs_error=error,loss_abs_error=abs(baseline0['loss']-changed['loss']),
            gradient_max_abs_error=max(float((grad[k]-newgrad[k]).abs().max()) for k in grad),
            original_repeat_gradient_max_abs_error=baseline_jitter,
            failed_gradient_names=failed,
            speedup=sum(r['seconds'] for r in baseline)/(changed['seconds']+more['seconds']))
        report['cases'].append(row);write_json(output,report);print(json.dumps(row),flush=True)
        if not diagnostic and (error>5e-6 or failed):raise AssertionError('Full model equivalence failed')
        del grad,newgrad,pred,newpred,captured
        if diagnostic:break
    report.update(status='diagnostic_only_not_acceptance' if diagnostic else 'passed',optimizer_steps=0)
    write_json(output,report)


if __name__=='__main__':main()
