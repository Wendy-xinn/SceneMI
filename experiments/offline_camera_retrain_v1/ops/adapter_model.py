"""EXO-gated low-rank updates inside an otherwise immutable native30 denoiser.

The original state-dict keys are retained.  A self-contained checkpoint stores
``model.state_dict()`` and ``adapter_spec(model)``; reconstruct the original
model, install the saved spec, and load the full state dict strictly.  This
module does not load observations or targets, sample motion, or select models.

Routing uses an explicit observation mode, never the number of visible joints.
Context variables scope adapter routing/disablement to one forward call, with
exception-safe restoration.  The underlying native30 model itself retains its
existing serial-forward requirement (temporary scene hooks and query caches).
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import math
from pathlib import Path
from threading import RLock
from types import MethodType

import torch
from torch import nn


PROTOCOL = "native30-exo-gated-internal-lora-v1"
STATE_PREFIX = "exo_adapters."


def _default_targets(model):
    """An explicit native30 target family; never adapt the final pose output."""
    unet = model.core.unet
    paths = ["core.unet.time_mlp.0", "core.unet.time_mlp.2"]
    blocks = [f"core.unet.downs.{i}.{j}" for i in range(len(unet.downs)) for j in (0, 1)]
    blocks += ["core.unet.mid_block1", "core.unet.mid_block2"]
    blocks += [f"core.unet.ups.{i}.{j}" for i in range(len(unet.ups)) for j in (0, 1)]
    for path in blocks:
        paths.extend((path + ".time_mlp.1", path + ".blocks.0.block1.0",
                      path + ".blocks.1.block.0"))
    paths.extend(("core.unet.final_conv.0.block.0", "observation_projection",
                  "camera_encoder.2", "reference_projection", "patch_condition",
                  "dynamic_condition", "region_condition"))
    return paths


def _canonical_spec(model, supplied):
    spec = copy.deepcopy(supplied) if supplied is not None else {}
    unknown = set(spec) - {"protocol", "rank", "alpha", "targets"}
    if unknown:
        raise ValueError("Unknown adapter options: " + ", ".join(sorted(unknown)))
    spec.setdefault("protocol", PROTOCOL)
    spec.setdefault("rank", 8)
    spec.setdefault("alpha", float(spec["rank"]))
    if spec["protocol"] != PROTOCOL:
        raise ValueError("Unknown adapter protocol")
    rank = spec["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("Adapter rank must be a positive integer")
    alpha = spec["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Adapter alpha must be positive and finite")
    spec["alpha"] = float(alpha)
    if "targets" not in spec:
        spec["targets"] = _default_targets(model)
    paths = spec["targets"]
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
        raise ValueError("Adapter targets must be a nonempty list of module paths")
    if len(paths) != len(set(paths)) or any(p.startswith(STATE_PREFIX) for p in paths):
        raise ValueError("Duplicate or recursive adapter targets")
    return spec


class _LowRankUpdate(nn.Module):
    """Linear/temporal-convolution weight update, with zero initial output."""
    def __init__(self, source, rank, alpha):
        super().__init__()
        self.scale = alpha / rank
        if isinstance(source, nn.Linear):
            self.down = nn.Linear(source.in_features, rank, bias=False)
            self.up = nn.Linear(rank, source.out_features, bias=False)
        elif isinstance(source, nn.Conv1d):
            if source.groups != 1 or source.padding_mode != "zeros":
                raise ValueError("Only ungrouped zero-padded Conv1d adapters are supported")
            self.down = nn.Conv1d(source.in_channels, rank, source.kernel_size,
                                  stride=source.stride, padding=source.padding,
                                  dilation=source.dilation, bias=False)
            self.up = nn.Conv1d(rank, source.out_channels, 1, bias=False)
        else:
            raise TypeError("Adapter target must be Linear or Conv1d")
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.up(self.down(value)) * self.scale


class _Routing:
    def __init__(self, original_forward, original_train, spec, paths, original_parameters):
        self.original_forward = original_forward
        self.original_train = original_train
        self.spec = spec
        self.paths = paths
        self.original_parameters = original_parameters
        self.lock = RLock()
        self.active = ContextVar("native30_exo_adapter_forward", default=None)
        self.disabled = ContextVar("native30_exo_adapter_disabled", default=False)
        self.handles = []


def _routing(model):
    routing = getattr(model, "_exo_adapter_routing", None)
    if routing is None:
        raise ValueError("No EXO adapters are installed")
    return routing


def _forward(self, *args, **kwargs):
    routing = _routing(self)
    batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
    noisy = kwargs.get("noisy_motion", args[0] if args else None)
    if not isinstance(batch, dict) or not torch.is_tensor(noisy) or noisy.ndim < 1:
        raise ValueError("Adapter forward requires noisy_motion and an explicit batch")
    modes = batch.get("mode")
    if not isinstance(modes, (tuple, list)) or len(modes) != noisy.shape[0]:
        raise ValueError("Explicit batch mode must contain one ego/exo entry per sample")
    if any(mode not in ("ego", "exo") for mode in modes):
        raise ValueError("Adapter mode must be ego or exo")
    # EGO-only forwards bypass every adapter operation.  Setting a local None
    # also prevents an outer EXO context from leaking into a nested EGO call.
    selected = tuple(i for i, mode in enumerate(modes) if mode == "exo")
    context = None if not selected or routing.disabled.get() else (len(modes), selected)
    # native30 already requires serial forwards (scene hooks/query caches).
    # Serialize this model explicitly while temporarily restoring inference
    # metadata.  CUDA cross-attention numerics can depend on weight
    # requires_grad flags even under no_grad; changing the global MHA fastpath
    # switch does not remove that difference.  Restore the original flags only
    # when no graph can be built, so inference retains the original 41k path.
    # Gradient-enabled adapter training keeps every original parameter frozen.
    # Its internal no_grad proposal also stays frozen; only top-level no_grad
    # inference restores metadata, avoiding parameter mutation mid-backprop.
    with routing.lock:
        restore_flags = not torch.is_grad_enabled()
        token = routing.active.set(context)
        try:
            if restore_flags:
                for parameter, original_flag in routing.original_parameters:
                    parameter.requires_grad_(original_flag)
            return routing.original_forward(*args, **kwargs)
        finally:
            if restore_flags:
                for parameter, _ in routing.original_parameters:
                    parameter.requires_grad_(False)
            routing.active.reset(token)


def _train(self, mode=True):
    if not isinstance(mode, bool):
        raise ValueError("Training mode must be boolean")
    routing = _routing(self)
    # Frozen weights alone do not disable scene-encoder dropout.  All original
    # modules stay in inference mode, while adapter parameters remain trainable.
    routing.original_train(False)
    self.exo_adapters.train(mode)
    return self


def _hook(routing, adapter):
    def apply(module, arguments, output):
        context = routing.active.get()
        if context is None:
            return None
        batch_size, selected = context
        if not arguments or not torch.is_tensor(arguments[0]) or not torch.is_tensor(output):
            raise ValueError("Adapter target requires tensor input/output")
        value = arguments[0]
        if value.shape[0] != batch_size or output.shape[0] != batch_size:
            raise ValueError("Selected adapter targets require batch axis zero")
        indices = torch.tensor(selected, device=value.device, dtype=torch.long)
        delta = adapter(value.index_select(0, indices))
        if delta.shape != output.index_select(0, indices).shape:
            raise ValueError("Adapter update does not match target output shape")
        # Select only EXO rows, rather than multiply an all-row update by zero:
        # even arbitrary nonfinite adapter values cannot alter an EGO row.
        updated = output.index_select(0, indices) + delta
        # Out-of-place index_copy always materializes a contiguous result.
        # Keep dense noncontiguous layouts too: changing a downstream kernel's
        # input strides can change its floating-point reduction, including on
        # EGO rows that were not updated.  Clone remains differentiable and
        # prevents the selected-row write from mutating the original output.
        result = output.clone(memory_format=torch.preserve_format)
        result.index_copy_(0, indices, updated)
        return result
    return apply


def install_adapters(model, spec=None):
    """Freeze the original model and install explicit-mode internal adapters.

    Selected targets must consume and produce tensors whose first axis is the
    motion batch.  The default list is specific to the current native30 UNet;
    custom explicit lists support small CPU tests and future declared variants.
    Validation completes before the model is mutated.
    """
    if hasattr(model, "_exo_adapter_routing") or hasattr(model, "exo_adapters"):
        raise ValueError("EXO adapters are already installed")
    spec = _canonical_spec(model, spec)
    modules = dict(model.named_modules())
    targets = []
    for path in spec["targets"]:
        if path not in modules:
            raise ValueError("Missing declared adapter target: " + path)
        target = modules[path]
        if not isinstance(target, (nn.Linear, nn.Conv1d)):
            raise TypeError("Adapter target must be Linear or Conv1d: " + path)
        if isinstance(target, nn.Conv1d) and (target.groups != 1 or target.padding_mode != "zeros"):
            raise ValueError("Unsupported grouped/padded Conv1d adapter target: " + path)
        targets.append(target)
    adapters = nn.ModuleDict({f"a{i:04d}": _LowRankUpdate(target, spec["rank"], spec["alpha"])
                              for i, target in enumerate(targets)})
    original_forward, original_train = model.forward, model.train
    original_parameters = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    for parameter, _ in original_parameters:
        parameter.requires_grad_(False)
    original_train(False)
    model.add_module("exo_adapters", adapters)
    routing = _Routing(original_forward, original_train, spec, list(spec["targets"]), original_parameters)
    model._exo_adapter_routing = routing
    for target, adapter in zip(targets, adapters.values()):
        routing.handles.append(target.register_forward_hook(_hook(routing, adapter)))
    model.forward = MethodType(_forward, model)
    model.train = MethodType(_train, model)
    model.train(True)
    return model


def adapter_spec(model):
    return copy.deepcopy(_routing(model).spec)


def adapter_parameters(model):
    _routing(model)
    return model.exo_adapters.parameters()


def base_state_dict(model):
    _routing(model)
    return OrderedDict((name, value) for name, value in model.state_dict().items()
                       if not name.startswith(STATE_PREFIX))


def adapter_state_dict(model):
    _routing(model)
    return model.exo_adapters.state_dict()


def _state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def base_state_digest(model):
    return _state_digest(base_state_dict(model))


def adapter_state_digest(model):
    return _state_digest(adapter_state_dict(model))


@contextmanager
def adapters_disabled(model):
    routing = _routing(model)
    token = routing.disabled.set(True)
    try:
        yield model
    finally:
        routing.disabled.reset(token)


def adapter_manifest(model):
    routing = _routing(model)
    modules = dict(model.named_modules())
    targets = []
    for index, path in enumerate(routing.paths):
        source = modules[path]
        entry = dict(path=path, adapter_key=f"a{index:04d}", type=type(source).__name__,
                     weight_shape=list(source.weight.shape),
                     parameters=sum(p.numel() for p in model.exo_adapters[f"a{index:04d}"].parameters()))
        targets.append(entry)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return dict(protocol=PROTOCOL, spec=adapter_spec(model), targets=targets,
                target_count=len(targets), trainable_parameters=sum(p.numel() for _, p in trainable),
                trainable_parameter_tensors=len(trainable),
                base_parameters=sum(p.numel() for name, p in model.named_parameters()
                                    if not name.startswith(STATE_PREFIX)),
                all_base_parameters_frozen=all(not p.requires_grad for name, p in model.named_parameters()
                                              if not name.startswith(STATE_PREFIX)),
                only_adapters_trainable=all(name.startswith(STATE_PREFIX) for name, _ in trainable),
                original_model_training=model.training,
                original_modules_all_eval=all(not module.training for name, module in model.named_modules()
                                             if name != "exo_adapters" and not name.startswith(STATE_PREFIX)),
                mode_source="explicit batch['mode']; no visibility-count inference",
                gate="EGO path bypass; EXO-only row selection; ContextVar-scoped restoration",
                inference_parameter_metadata="original requires_grad flags only during serialized no_grad forward; finally frozen",
                training_parameter_metadata="all original parameters frozen, including internal no_grad proposal",
                deployment="one state_dict containing frozen parent and internal adapters; save spec for strict reload")


def load_inference_model(checkpoint_path, device="cpu", expected_sha256=None):
    """Load one complete frozen-base/adapter checkpoint without its parent.

    Returns ``(model, metadata)``.  The embedded configuration constructs the
    architecture only: this function never follows its warm-start path, opens a
    dataset, or constructs observations.  Body/scene observations are supplied
    separately through the normal native30 inference interface.
    """
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model

    path = Path(checkpoint_path).expanduser().resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("Inference checkpoint SHA256 differs from the declared file")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Inference checkpoint must contain a complete state dictionary")
    required = {"model", "effective_config", "adapter_spec", "frozen_base_sha256", "adapter_sha256",
                "step", "planned_steps"}
    if not required <= state.keys():
        raise ValueError("Inference checkpoint lacks embedded base/adapter identity or configuration")
    config = state["effective_config"].get("config")
    if not isinstance(config, dict) or config.get("diffusion_target") != "reference_residual":
        raise ValueError("Inference checkpoint is not the native30 residual diffusion family")
    step, planned = state["step"], state["planned_steps"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (step, planned)) or not 0 <= step <= planned:
        raise ValueError("Invalid inference checkpoint stage counters")
    spec = state["adapter_spec"]
    if not isinstance(spec, dict):
        raise ValueError("Inference checkpoint adapter specification is missing")
    for key, value in config.get("adapters", {}).items():
        if spec.get(key) != value:
            raise ValueError("Embedded configuration and checkpoint adapter specification disagree: " + key)
    if not isinstance(state["model"], dict):
        raise ValueError("Inference checkpoint has no complete model state")
    for name, value in state["model"].items():
        if not torch.is_tensor(value) or value.device.type != "cpu":
            raise ValueError("Invalid inference model state tensor: " + name)
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite inference model state tensor: " + name)
    model = make_model(config)
    install_adapters(model, spec)
    model.load_state_dict(state["model"], strict=True)
    base_hash, adapter_hash = base_state_digest(model), adapter_state_digest(model)
    if base_hash != state["frozen_base_sha256"]:
        raise ValueError("Inference frozen-base digest differs from the saved checkpoint identity")
    if adapter_hash != state["adapter_sha256"]:
        raise ValueError("Inference adapter digest differs from the saved checkpoint identity")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Inference checkpoint changed during loading")
    model.to(device).eval()
    metadata = dict(checkpoint=str(path), checkpoint_sha256=checksum, checkpoint_bytes=before.st_size,
                    config=copy.deepcopy(config), adapter_spec=adapter_spec(model), stage_step=step,
                    planned_stage_updates=planned,
                    parent_step=state.get("initialization", {}).get("parent_step", config.get("warm_start", {}).get("step")),
                    frozen_base_sha256=base_hash, adapter_sha256=adapter_hash,
                    strict_full_state_loaded=True, all_state_tensors_finite=True,
                    parent_checkpoint_opened=False, dataset_opened=False,
                    device=str(next(model.parameters()).device))
    return model, metadata
