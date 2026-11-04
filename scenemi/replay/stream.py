"""Matched, source-stratified training examples independent of optimizer batch."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.training import now
from .data import CORPORA

SOURCE_WEIGHTS=(.25,.125,.28125,.34375)


def stream_requests(seed,start,count):
    if start%32 or count%32:raise ValueError('Aligned blocks of 32 windows required')
    result=[]
    for group in range(start//32,(start+count)//32):
        # Four consecutive 32-window blocks have exactly the batch128 quotas.
        counts=(4,2,4,6) if group%2==0 else (4,2,5,5)
        sources=np.repeat(np.arange(4),counts)
        np.random.default_rng(np.random.SeedSequence([seed,group,0])).shuffle(sources)
        for i,source in enumerate(sources):
            for k,mode in enumerate(('ego','exo')):
                index=group*32+2*i+k
                sample_seed=int(np.random.default_rng(np.random.SeedSequence([seed,index,1])).integers(0,2**63-1))
                result.append(dict(position=index,dataset=CORPORA[int(source)],mode=mode,seed=sample_seed,step=1000))
    return result


def add_noise(pair,request,seed):
    sample,identity=pair;rng=np.random.default_rng(np.random.SeedSequence([seed,request['position'],2]))
    sample=dict(sample,diffusion_t=np.int64(rng.integers(1000 if request['mode']=='ego' else 201)),
        diffusion_noise=rng.standard_normal((128,201),dtype=np.float32))
    return sample,dict(identity,stream_index=request['position'],stream_sample_seed=request['seed'])


def build_cache(sampler,folder,windows,seed,workers=4):
    folder.mkdir(parents=True,exist_ok=True);manifest=[]
    if windows%128:raise ValueError('Cache budget must align with 128-window source blocks')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit(start):
            req=stream_requests(seed,start,128)
            return req,[pool.submit(sampler.sample,r) for r in req]
        pending=submit(0)
        for start in range(0,windows,128):
            req,futures=pending;pairs=[add_noise(f.result(),r,seed) for f,r in zip(futures,req)]
            if start+128<windows:pending=submit(start+128)
            data=dict(ego=collate([p[0] for p in pairs[::2]]),exo=collate([p[0] for p in pairs[1::2]]),identities=[p[1] for p in pairs])
            path=folder/f'{start//128:04d}.pt';temporary=path.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(path)
            manifest.append(dict(path=str(path),sha256=sha256(path),start=start,windows=128))
            if start%1024==0:atomic_json(folder/'STATUS.json',dict(state='preparing',completed=start+128,total=windows,updated_at_utc=now()))
    result=dict(state='complete',windows=windows,seed=seed,shards=manifest,source_weights=SOURCE_WEIGHTS,test_used=False,
        labels_never_in_model_payload=True,matched_perturbations_and_diffusion_noise=True)
    atomic_json(folder/'MANIFEST.json',result);return result


class CachedStream:
    def __init__(self,manifest):
        self.manifest=manifest;self.pool=ThreadPoolExecutor(max_workers=1);self.pending={};self.current=None
        self.submit(0)
    def submit(self,block):
        if block<len(self.manifest['shards']) and block not in self.pending:self.pending[block]=self.pool.submit(self.load,block)
    def load(self,block):
        row=self.manifest['shards'][block];path=Path(row['path'])
        if sha256(path)!=row['sha256']:raise ValueError('Matched training cache changed')
        return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    def get(self,start,batch,device='cuda'):
        if batch not in (32,128) or start%batch:raise ValueError('Unsupported paired optimizer batch')
        block=start//128
        if self.current is None or self.current[0]!=block:
            self.submit(block);self.current=block,self.pending.pop(block).result();self.submit(block+1)
        data=self.current[1];first=(start%128)//2;last=first+batch//2
        values=[]
        for mode in ('ego','exo'):
            chunk={k:v[first:last] for k,v in data[mode].items()}
            if str(device).startswith('cuda'):chunk={k:v.pin_memory().to(device,non_blocking=True) for k,v in chunk.items()}
            values.append(chunk)
        return *values,data['identities'][first*2:last*2]
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
