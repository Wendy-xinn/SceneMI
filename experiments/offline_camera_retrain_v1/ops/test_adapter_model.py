"""CPU invariants for explicit-mode adaptation inside a frozen denoiser."""
import copy
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .adapter_model import (
    adapter_manifest, adapter_parameters, adapter_spec, adapter_state_digest,
    adapters_disabled, base_state_dict, base_state_digest, install_adapters, load_inference_model,
)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.temporal = nn.Conv1d(4, 4, 3, padding=1)
        self.dropout = nn.Dropout(.8)

    def forward(self, noisy_motion, timestep, batch, *, return_details=False):
        if batch.get("barrier") is not None:
            batch["barrier"].wait(timeout=5)
        with torch.no_grad():
            proposal = self.projection(noisy_motion)
        if batch.get("fail"):
            raise RuntimeError("deliberate forward failure")
        raw = self.projection(noisy_motion)
        result = self.temporal(self.dropout(raw).transpose(1, 2)).transpose(1, 2) + .1 * proposal
        return (result, dict(raw_motion=raw, proposal=proposal.detach())) if return_details else result


SPEC = dict(rank=2, alpha=2, targets=["projection", "temporal"])


class AdapterModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(527)
        self.base = TinyDenoiser().eval()
        self.x = torch.randn(2, 7, 4)
        self.t = torch.tensor([20, 500])

    def model(self):
        return install_adapters(copy.deepcopy(self.base), SPEC)

    def perturb(self, model):
        with torch.no_grad():
            for adapter in model.exo_adapters.values():
                adapter.up.weight.normal_(0, .3)

    def test_zero_initialization_preserves_both_modes_and_details(self):
        model = self.model()
        for modes in (["ego", "ego"], ["exo", "exo"], ["ego", "exo"]):
            expected, old = self.base(self.x, self.t, dict(mode=modes), return_details=True)
            actual, new = model(self.x, self.t, dict(mode=modes), return_details=True)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(old["raw_motion"], new["raw_motion"]))
            self.assertTrue(torch.equal(old["proposal"], new["proposal"]))

    def test_updated_adapters_change_exo_but_preserve_ego_and_mixed_rows(self):
        model = self.model()
        self.perturb(model)
        expected = self.base(self.x, self.t, dict(mode=["ego", "ego"]))
        ego = model(self.x, self.t, dict(mode=["ego", "ego"]))
        exo = model(self.x, self.t, dict(mode=["exo", "exo"]))
        mixed = model(self.x, self.t, dict(mode=["ego", "exo"]))
        self.assertTrue(torch.equal(expected, ego))
        self.assertFalse(torch.equal(expected, exo))
        self.assertTrue(torch.equal(expected[0], mixed[0]))
        # EXO adapter matmuls see a different selected-batch size (two versus
        # one); their floating-point reduction order need not be bit-identical.
        torch.testing.assert_close(exo[1], mixed[1], rtol=1e-6, atol=1e-7)
        with adapters_disabled(model):
            self.assertTrue(torch.equal(expected, model(self.x, self.t, dict(mode=["exo", "exo"]))))
        self.assertTrue(torch.equal(exo, model(self.x, self.t, dict(mode=["exo", "exo"]))))

    def test_only_adapters_receive_gradients_and_base_stays_eval_and_unchanged(self):
        model = self.model()
        before = base_state_digest(model)
        model.train()
        manifest = adapter_manifest(model)
        self.assertTrue(manifest["all_base_parameters_frozen"])
        self.assertTrue(manifest["original_modules_all_eval"])
        self.assertTrue(manifest["only_adapters_trainable"])
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=.03)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            actual, details = model(self.x, self.t, dict(mode=["ego", "exo"]), return_details=True)
            self.assertFalse(details["proposal"].requires_grad)
            self.assertTrue(details["raw_motion"].requires_grad)
            actual[1].square().mean().backward()
            grads = [p.grad for p in adapter_parameters(model) if p.grad is not None]
            self.assertTrue(any(g.abs().sum() > 0 for g in grads))
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            optimizer.step()
        self.assertEqual(before, base_state_digest(model))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if not name.startswith("exo_adapters.")))
        expected = self.base(self.x, self.t, dict(mode=["ego", "ego"]))
        self.assertTrue(torch.equal(expected, model(self.x, self.t, dict(mode=["ego", "ego"]))))
        model.eval()
        self.assertFalse(model.exo_adapters.training)
        model.train()
        self.assertTrue(model.exo_adapters.training)
        self.assertFalse(model.training)

    def test_full_state_strict_reload_preserves_original_keys(self):
        model = self.model()
        self.perturb(model)
        saved = copy.deepcopy(model.state_dict())
        self.assertEqual(set(base_state_dict(model)), set(self.base.state_dict()))
        other = install_adapters(copy.deepcopy(self.base), adapter_spec(model))
        result = other.load_state_dict(saved, strict=True)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        self.assertEqual(adapter_state_digest(model), adapter_state_digest(other))
        for modes in (["ego", "ego"], ["exo", "exo"], ["ego", "exo"]):
            self.assertTrue(torch.equal(model(self.x, self.t, dict(mode=modes)),
                                        other(self.x, self.t, dict(mode=modes))))

    def test_invalid_mode_is_rejected_and_exception_does_not_leak_routing(self):
        model = self.model()
        self.perturb(model)
        for modes in (None, "exo", ["exo"], ["exo", "unknown"]):
            with self.assertRaises(ValueError):
                model(self.x, self.t, dict(mode=modes))
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            model(self.x, self.t, dict(mode=["exo", "exo"], fail=True))
        expected = self.base(self.x, self.t, dict(mode=["ego", "ego"]))
        self.assertTrue(torch.equal(expected, model(self.x, self.t, dict(mode=["ego", "ego"]))))
        self.assertTrue(torch.equal(self.base.projection(self.x), model.projection(self.x)))

    def test_routing_is_isolated_between_concurrent_forward_contexts(self):
        model = self.model()
        self.perturb(model)
        expected_ego = model(self.x, self.t, dict(mode=["ego", "ego"]))
        expected_exo = model(self.x, self.t, dict(mode=["exo", "exo"]))
        barrier = Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            ego = pool.submit(model, self.x, self.t, dict(mode=["ego", "ego"], barrier=barrier))
            exo = pool.submit(model, self.x, self.t, dict(mode=["exo", "exo"], barrier=barrier))
            self.assertTrue(torch.equal(expected_ego, ego.result(timeout=10)))
            self.assertTrue(torch.equal(expected_exo, exo.result(timeout=10)))

    def test_target_validation_happens_before_base_is_frozen(self):
        for spec in (dict(SPEC, targets=["missing"]), dict(SPEC, targets=["dropout"]),
                     dict(SPEC, rank=0), dict(SPEC, targets=["projection", "projection"])):
            model = copy.deepcopy(self.base)
            with self.assertRaises((ValueError, TypeError)):
                install_adapters(model, spec)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertFalse(hasattr(model, "exo_adapters"))

    def inference_checkpoint(self, model):
        return dict(model=copy.deepcopy(model.state_dict()),
                    effective_config=dict(config=dict(diffusion_target="reference_residual",
                        adapters=adapter_spec(model),
                        warm_start=dict(checkpoint="/nonexistent-parent-must-not-be-opened.pt", step=41000))),
                    adapter_spec=adapter_spec(model), frozen_base_sha256=base_state_digest(model),
                    adapter_sha256=adapter_state_digest(model), step=1000, planned_steps=1000)

    def test_standalone_inference_loader_needs_no_parent_or_dataset(self):
        model = self.model()
        self.perturb(model)
        saved = self.inference_checkpoint(model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "standalone.pt"
            torch.save(saved, path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("experiments.offline_camera_retrain_v1.prepare_native30.make_model",
                       side_effect=lambda config: TinyDenoiser()) as factory:
                loaded, meta = load_inference_model(path, expected_sha256=checksum)
            factory.assert_called_once_with(saved["effective_config"]["config"])
            self.assertEqual(meta["checkpoint_sha256"], checksum)
            self.assertEqual(meta["parent_step"], 41000)
            self.assertEqual(meta["stage_step"], 1000)
            self.assertFalse(meta["parent_checkpoint_opened"] or meta["dataset_opened"])
            self.assertTrue(meta["all_state_tensors_finite"])
            self.assertTrue(adapter_manifest(loaded)["original_modules_all_eval"])
            self.assertFalse(loaded.exo_adapters.training)
            for modes in (["ego", "ego"], ["exo", "exo"], ["ego", "exo"]):
                self.assertTrue(torch.equal(model(self.x, self.t, dict(mode=modes)),
                                            loaded(self.x, self.t, dict(mode=modes))))

    def test_inference_loader_rejects_wrong_checksum_and_tensor_identity(self):
        model = self.model()
        saved = self.inference_checkpoint(model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "standalone.pt"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_inference_model(path, expected_sha256="0" * 64)
            saved["frozen_base_sha256"] = "0" * 64
            torch.save(saved, path)
            with patch("experiments.offline_camera_retrain_v1.prepare_native30.make_model",
                       side_effect=lambda config: TinyDenoiser()):
                with self.assertRaisesRegex(ValueError, "frozen-base digest"):
                    load_inference_model(path)
            saved["model"]["exo_adapters.a0000.up.weight"][0, 0] = float("nan")
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                load_inference_model(path)


if __name__ == "__main__":
    unittest.main()
