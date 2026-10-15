"""Bounded task-owned verified read cache; native source files never removed."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array


class StreamData(Native30Data):
    def __init__(self,*args,read_cache,**kwargs):
        self.task_read_cache=Path(read_cache);self.task_read_cache.mkdir(exist_ok=True)
        self.task_read_limit=4*2**30
        self.read_lru=OrderedDict((p,p.stat().st_size) for p in sorted(self.task_read_cache.glob('*.npy'),key=lambda p:p.stat().st_mtime))
        super().__init__(*args,**kwargs)

    def load(self,folder,key):
        source=Path(folder)/f'{key}.npy'
        expected=self.source_hashes[str(folder)][key+'.npy']
        global_cache=Path(__file__).resolve().parents[2]/'data/native30_verified_read_cache_v1'
        # Existing immutable mirrors are shared. New potentially large EXO
        # arrays use only this task's4GiB LRU, avoiding an unbounded Linux mirror.
        if (global_cache/(expected+'.npy')).exists() or not str(source.resolve()).startswith('/mnt/'):
            return super().load(folder,key)
        target=self.task_read_cache/(expected+'.npy')
        if not target.exists():
            incoming=source.stat().st_size
            if incoming>self.task_read_limit:raise OSError('Single source array exceeds task verified-cache limit')
            while self.read_lru and sum(self.read_lru.values())+incoming>self.task_read_limit:
                path,_=self.read_lru.popitem(last=False)
                assert path.parent==self.task_read_cache and len(path.stem)==64
                path.unlink(missing_ok=True)
        result=verified_array(source,expected,self.task_read_cache)
        self.read_lru[target]=target.stat().st_size;self.read_lru.move_to_end(target)
        # Do not retain mmap handles in data.arrays: unlinked private cache
        # files must release their disk pages when the current preparation ends.
        return result
