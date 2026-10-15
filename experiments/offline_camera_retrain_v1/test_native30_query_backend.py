"""Backend isolation across models, exceptions, deepcopy and state dictionaries."""
import copy
import unittest
import torch
from experiments.offline_camera_retrain_v1.native30_query_backend import attach_query_backend
from experiments.offline_camera_retrain_v1 import native30_query_runtime as runtime
from experiments.offline_camera_retrain_v1.native30_exact_query import query_surfaces_cuda


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
    def forward(self,x,fail=False):
        if fail:raise RuntimeError('probe')
        return self.weight*x,runtime.query_surfaces_cuda


class QueryBackendTests(unittest.TestCase):
    def test_opt_in_is_model_scoped_and_has_no_checkpoint_keys(self):
        old=runtime.query_surfaces_cuda;model=Probe();keys=set(model.state_dict())
        attach_query_backend(model,{'runtime':{'query_backend':'cuda_exact_radius_v1'}})
        self.assertEqual(set(model.state_dict()),keys)
        self.assertIs(model(torch.ones(()))[1],query_surfaces_cuda)
        self.assertIs(runtime.query_surfaces_cuda,old)
        self.assertIs(Probe()(torch.ones(()))[1],old)
        attach_query_backend(model,{'runtime':{'query_backend':'reference'}})
        self.assertIs(model(torch.ones(()))[1],old)
        attach_query_backend(model,{'runtime':{'query_backend':'cuda_exact_radius_v1'}})
        self.assertIs(model(torch.ones(()))[1],query_surfaces_cuda)
        self.assertIs(runtime.query_surfaces_cuda,old)
    def test_exception_restores_global_backend(self):
        old=runtime.query_surfaces_cuda
        model=attach_query_backend(Probe(),{'runtime':{'query_backend':'cuda_exact_radius_v1'}})
        with self.assertRaisesRegex(RuntimeError,'probe'):model(torch.ones(()),fail=True)
        self.assertIs(runtime.query_surfaces_cuda,old)
    def test_deepcopy_forward_uses_copied_model(self):
        model=attach_query_backend(Probe(),{'runtime':{'query_backend':'cuda_exact_radius_v1'}})
        other=copy.deepcopy(model)
        with torch.no_grad():other.weight.fill_(3)
        self.assertEqual(float(other(torch.ones(()))[0]),3)
        self.assertEqual(float(model(torch.ones(()))[0]),1)
    def test_default_backend_preserves_forward(self):
        model=Probe();forward=model.forward
        self.assertIs(attach_query_backend(model,{}),model)
        self.assertEqual(model.forward,forward)
        with self.assertRaises(ValueError):attach_query_backend(model,{'runtime':{'query_backend':'unknown'}})
