"""Counter-seeded replay batches, checked read caches and ordered CPU prefetch."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import threading

import numpy as np
import torch

from scenemi.data.amass_dataset import collate
from scenemi.data.exo_initialization import CONDITIONS,repair_example

CORPORA=('amass','egobody','trumans','grail')


class ArrayCache:
    """Verify on first read, then reject changed mapped files before reuse."""
    def __init__(self,loader,size=32):self.loader=loader;self.size=size;self.cache=OrderedDict()

    def __call__(self,meta,name):
        key=meta['path'],name,meta['output_sha256'][name+'.npy']
        if key not in self.cache:
            value=self.loader(meta,name);path=Path(value.filename);stat=path.stat()
            self.cache[key]=(value,path,(stat.st_size,stat.st_mtime_ns,stat.st_ino))
            while len(self.cache)>self.size:self.cache.popitem(last=False)
        value,path,expected=self.cache[key];stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=expected:raise ValueError('Verified mapped data changed: '+str(path))
        self.cache.move_to_end(key);return value


def clone_dataset(data):
    value=copy.copy(data)
    if hasattr(data,'selection_bundle'):
        value.selection_bundle=copy.copy(data.selection_bundle)
        value.selection_bundle.array=ArrayCache(data.selection_bundle.array)
        # Use one bounded cache per worker/mode; nested retained mmaps both
        # bypass freshness checks and can exhaust the service's descriptor cap.
        value.array=value.selection_bundle.array
        value.array_cache=OrderedDict();value.prefix_cache=OrderedDict()
    if hasattr(data,'bundle'):
        value.bundle=copy.copy(data.bundle);value.bundle.array=ArrayCache(data.bundle.array)
    return value


def requests(seed,step,batch):
    if batch<8 or batch%8:raise ValueError('Replay batch must divide into four sources and two modes')
    rng=np.random.default_rng(np.random.SeedSequence([seed,step]))
    sources=np.tile(np.arange(4),batch//8);rng.shuffle(sources)
    return [dict(position=i*2+k,dataset=CORPORA[int(source)],mode=mode,
        seed=int(rng.integers(0,2**63-1)),step=step)
        for i,source in enumerate(sources) for k,mode in enumerate(('ego','exo'))]


def normalize(sample,identity,dataset,mode):
    sample=dict(sample);identity=dict(identity,source_dataset=dataset)
    if dataset=='amass':
        sample.update(body_format=np.int64(0),sensor_profile=np.int64(0),
            target_native_betas=np.zeros(10,np.float32),target_template=np.int64(0))
    else:
        code=3 if dataset=='grail' else (1 if identity.get('native_format')=='smpl' else 2)
        if mode=='ego':code=int(sample['body_format'])
        native_beta=sample.get('target_native_betas',sample.get('target_betas',np.zeros(10,np.float32)))
        sample.update(body_format=np.int64(code),sensor_profile=np.int64(sample.get('sensor_profile',0)),
            target_native_betas=np.asarray(native_beta,np.float32),
            target_template=np.int64(sample.get('target_template',0)),target_betas=np.zeros(16,np.float32))
    sample['dataset_id']=np.int64(CORPORA.index(dataset))
    return sample,identity


class ReplaySampler:
    def __init__(self,amass,native,frontend,seed):
        if amass.split!='train' or any(d.split!='train' for d in native.values()):
            raise ValueError('Replay can only draw from training splits')
        self.amass=amass;self.native=native;self.frontend=frontend;self.seed=seed;self.local=threading.local()

    def sample(self,request):
        if not hasattr(self.local,'data'):
            self.local.data={m:clone_dataset(d) for m,d in self.native.items()}
        rng=np.random.default_rng(request['seed']);mode=request['mode'];source=request['dataset']
        condition=CONDITIONS[request['position']//2%4] if request['step']<=2 else None
        if source=='amass':
            if mode=='ego':sample,identity=self.amass.sample(rng,'ego',temporal_dropout=None)
            else:sample,identity=repair_example(self.amass,rng,self.frontend,condition=condition)
        else:
            data=self.local.data[mode];family=str(rng.choice(data.families[source]))
            member=int(rng.choice(data.groups[source][family]))
            kwargs=dict(member=member)
            if mode=='exo':sample,identity=data.sample(rng,self.frontend,condition=condition,**kwargs)
            else:sample,identity=data.sample(rng,**kwargs)
        return normalize(sample,identity,source,mode)


class Prefetch:
    """One ordered CPU batch ahead; step seeds make resume worker-independent."""
    def __init__(self,sampler,batch,workers=4):
        self.sampler=sampler;self.batch=batch;self.pool=ThreadPoolExecutor(max_workers=workers)
        self.pending={}

    def submit(self,step):
        if step not in self.pending:
            self.pending[step]=[self.pool.submit(self.sampler.sample,r) for r in requests(self.sampler.seed,step,self.batch)]

    def get(self,step,device='cpu',next_step=None):
        self.submit(step);pairs=[f.result() for f in self.pending.pop(step)]
        if next_step is not None:self.submit(next_step)
        result=[]
        for offset in [0,1]:
            values=collate([p[0] for p in pairs[offset::2]],'cpu')
            if str(device).startswith('cuda'):values={k:v.pin_memory().to(device,non_blocking=True) for k,v in values.items()}
            result.append(values)
        return *result,[p[1] for p in pairs]

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
