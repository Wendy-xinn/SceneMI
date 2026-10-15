"""B2/B4/B8 throughput/memory with Adam-sized buffers, zero weight updates."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile_full import *
from exact_gpu_query import query_surfaces_cuda
import experiments.offline_camera_retrain_v1.native30_query_runtime as runtime


def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.5)
    config=read_config(HERE/'native30_fresh_ego_config.json')
    cache=torch.load(ROOT/'timing_bank.pt',map_location='cuda',weights_only=False)
    assert cache['config']==config and cache['data_audit_sha256']==sha256(ROOT/'DATA_AUDIT.json')
    bank=cache['bank'];del cache
    source=HERE/'runs/native30_fresh_ego_oct14/training/last.pt'
    parent=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    model=make_model(config).cuda();model.load_state_dict(parent['model'],strict=True)
    weights=parent['effective_config']['observation_coefficients'];del parent
    # Only resident zero buffers; Adam.step is never executed.
    buffers=[torch.zeros_like(p) for p in model.parameters() for _ in range(2)]
    runtime.query_surfaces_cuda=query_surfaces_cuda
    report=dict(status='running',optimizer_steps=0,model_weight_updates=0,
        scope='Forward/objective/backward only; Adam-sized memory buffers included; preparation, actual Adam compute, checkpoint and validation excluded. Repeated timing windows are not quality experiments.',
        resident_moment_buffer_gib=sum(x.numel()*x.element_size() for x in buffers)/2**30,
        cuda_memory_fraction=.5,cases=[])
    write_json(ROOT/'BATCH_THROUGHPUT.json',report)
    for size in (2,4,8):
        group_order=['trumans','camera_wearer']*(size//2)
        batch=collate_observations([bank[k] for k in group_order]);known=model_inputs(config,batch)
        t=torch.full((size,),300,device='cuda',dtype=torch.long);signal=model.alphas[t]
        target=training_motion(config,batch,known)
        gen=torch.Generator(device='cuda').manual_seed(6621)
        noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn(target.shape,device='cuda',generator=gen)
        rows=[]
        for repeat in range(3):
            model.train();model.zero_grad(set_to_none=True);torch.manual_seed(10047)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
            pred,details=model(noisy,t,known,use_scene=True,return_details=True)
            loss,_=objective(pred,batch['motion'],batch,signal,details,config,weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite batch benchmark')
            loss.backward();torch.cuda.synchronize();elapsed=time.monotonic()-began
            rows.append(dict(seconds=elapsed,samples_per_second=size/elapsed,
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            del pred,details,loss
        row=dict(batch_size=size,warmup=rows[0],measured=rows[1:])
        report['cases'].append(row);write_json(ROOT/'BATCH_THROUGHPUT.json',report)
        print(json.dumps(row),flush=True)
        del batch,known,target,noisy
    report['status']='completed';write_json(ROOT/'BATCH_THROUGHPUT.json',report)


if __name__=='__main__':main()
