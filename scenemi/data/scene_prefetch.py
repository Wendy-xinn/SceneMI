"""One ordered scene-mode batch ahead, with a separate speculative RNG.

Consumers commit the RNG state after the last sample they actually consume.
Prefetched future samples never advance the checkpoint's committed stream.
"""
from concurrent.futures import ThreadPoolExecutor
import copy


class ModeDataset:
    def __init__(self,owner,mode):self.owner,self.mode=owner,mode
    def sample(self,rng,**kwargs):return self.owner.take(self.mode,rng)


class ScenePrefetch:
    def __init__(self,data,rng,counts,policy,max_batches=None):
        self.data=data;self.rng=copy.deepcopy(rng);self.counts=counts;self.policy=policy
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='scene-input')
        self.max_batches=max_batches;self.submitted=1
        self.mode='ego';self.index=0;self.batch=None
        self.future=self.pool.submit(self.prepare,'ego')
        self.datasets={mode:ModeDataset(self,mode) for mode in ('ego','exo')}

    def prepare(self,mode):
        pairs=[self.data[mode].sample(self.rng,**({'policy':self.policy} if mode=='exo' else {}))
               for _ in range(self.counts[mode])]
        return pairs,copy.deepcopy(self.rng.bit_generator.state)

    def take(self,mode,rng):
        if mode!=self.mode:raise ValueError('Scene-mode consumption order changed')
        if self.batch is None:
            self.batch,self.end_state=self.future.result()
            following='exo' if mode=='ego' else 'ego'
            if self.max_batches is None or self.submitted<self.max_batches:
                self.future=self.pool.submit(self.prepare,following);self.submitted+=1
            else:self.future=None
        value=self.batch[self.index];self.index+=1
        if self.index==len(self.batch):
            rng.bit_generator.state=copy.deepcopy(self.end_state)
            self.batch=None;self.index=0;self.mode='exo' if mode=='ego' else 'ego'
        return value

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
