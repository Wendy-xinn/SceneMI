"""Model-scoped exact query backend, preserving CPU supervision cache sources.

The existing surface API returns NumPy fields. Select the optimized CUDA path
only during this model's forward and restore the original on exit/exception.
As with the model's temporary core hooks, calls in one process are sequential.
"""
import functools
from types import MethodType

BACKENDS=('reference','cuda_exact_radius_v1')


def attach_query_backend(model,config):
    backend=config.get('runtime',{}).get('query_backend','reference')
    if backend not in BACKENDS:raise ValueError('Unknown scene query backend: '+str(backend))
    if hasattr(model,'_reference_query_forward'):
        model.query_backend=backend
        return model
    if backend=='reference':return model
    from experiments.offline_camera_retrain_v1 import native30_query_runtime as runtime
    from experiments.offline_camera_retrain_v1.native30_exact_query import query_surfaces_cuda
    original=model.forward
    model._reference_query_forward=original
    @functools.wraps(original)
    def forward(self,*args,**kwargs):
        if self.query_backend=='reference':return self._reference_query_forward(*args,**kwargs)
        previous=runtime.query_surfaces_cuda
        runtime.query_surfaces_cuda=query_surfaces_cuda
        try:return self._reference_query_forward(*args,**kwargs)
        finally:runtime.query_surfaces_cuda=previous
    model.forward=MethodType(forward,model)
    model.query_backend=backend
    return model
