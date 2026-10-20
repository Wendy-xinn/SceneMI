"""Prepare and serve complete GRAIL source reconstructions with MotionViewer.

This is a data inspection entry point, not model evaluation. It preserves all
saved source frames and uses the export's cameras and first-hit samples. The
source mesh is shared between observer panels, with no floor snap, fitting,
prediction, diffusion inference, or additional visibility rendering.

Usage::

    python -m experiments.offline_camera_retrain_v1.ops.grail_preview prepare \
        --manifest runs/grail_data_v1/pilot/manifest.jsonl --output runs/grail_data_v1/preview
    python -m experiments.offline_camera_retrain_v1.ops.grail_preview serve \
        --manifest runs/grail_data_v1/preview/PREPARED_CASES.json --port 8840

Paths are relative to the current working directory, not the examples above.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np

from .grail_source import G1_TEMPLATE, body_model_factory, file_sha256, g1_template_assets
from .motion_viewer import MotionViewer, atomic_json
from .viewer_geometry import camera_layers, instant_scene_layers, readonly_array

SCHEMA = "grail-source-preview-v1"
SOURCE_LABEL = "原始重建（未生成）"
WARNING = "GRAIL 原始重建数据预览，非模型预测"


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _source_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No completed GRAIL bundles in manifest")
    seen = set()
    for row in rows:
        folder = Path(row["path"])
        if not folder.is_absolute():
            folder = (path.parent / folder).resolve()
        metadata_path = folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if "metadata_sha256" in row:
            if set(row) != {"path", "metadata_sha256"} or row["metadata_sha256"] != file_sha256(metadata_path):
                raise ValueError(f"Compact source manifest metadata SHA256 mismatch: {folder}")
        elif {key: value for key, value in row.items() if key != "path"} != metadata:
            raise ValueError(f"Source manifest differs from completed metadata: {folder}")
        row.update(metadata)
        row["path"] = str(folder)
        row["metadata_sha256"] = file_sha256(metadata_path)
        if row.get("dataset") != "grail" or row.get("fps") != 30:
            raise ValueError("This preview only supports exported 30 Hz GRAIL source bundles")
        if row["sequence_id"] in seen:
            raise ValueError("A source sequence must be one complete case, without duplicate windows")
        seen.add(row["sequence_id"])
    return rows


def _full_scene(identity: dict) -> dict:
    vertices, faces, offset = [], [], 0
    for name in ("object", "ground"):
        v = np.asarray(readonly_array(identity, "scene_" + name + "_vertices"), np.float32)
        f = np.asarray(readonly_array(identity, "scene_" + name + "_faces"), np.int32)
        if v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,):
            raise ValueError("Malformed saved scene mesh")
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise ValueError("Saved scene mesh face index out of range")
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return dict(scene_vertices=np.concatenate(vertices), scene_faces=np.concatenate(faces))


def _observations(identity: dict, n: int, joints: np.ndarray) -> tuple[dict, dict]:
    """Expose actual saved conditions; never synthesize hidden joint positions."""
    from experiments.offline_camera_retrain_v1.data import rotation6d
    from experiments.offline_camera_retrain_v1.native30_observations import inverse_head
    from .viewer_observation import exo_first_hit_evidence

    observation = np.zeros((n, 22, 9), np.float32)
    metadata = np.zeros((n, 22, 5), np.float32)
    metadata[..., 4] = 1
    occluded = np.zeros((n, 22), bool)
    if identity["mode"] == "ego":
        position = readonly_array(identity, "camera_position_scenemi_yup")
        rotation = readonly_array(identity, "camera_rotation_scenemi_yup")
        camera = np.concatenate((position / 2, rotation6d(rotation)), -1)
        head, available, confidence = inverse_head(camera, np.ones((n, 2)),
            np.ones((n, 2)), identity["actual_virtual_mount"])
        observation[:, 15] = head
        metadata[:, 15, :2] = available
        metadata[:, 15, 2:4] = confidence
        error = float(np.max(np.abs(head[:, :3] * 2 - joints[:, 15])))
        if error > 1e-5:
            raise ValueError(f"Exported EGO camera mount does not reproduce source head: {error}")
        note = dict(kind="saved_virtual_head_camera_inverse_fixed_mount", head_position_max_error_m=error,
            occlusion_evidence="EGO head availability is not a body visibility/occlusion mask")
    else:
        available = np.asarray(readonly_array(identity, "exo_position_available"), np.float32)
        confidence = np.asarray(readonly_array(identity, "exo_position_confidence"), np.float32)
        trajectory = np.asarray(readonly_array(identity, "exo_trajectory_world"), np.float32)
        valid = readonly_array(identity, "observation_valid")
        evidence = exo_first_hit_evidence(readonly_array(identity, "exo_reference_counts"),
            readonly_array(identity, "exo_visible_counts"),
            {key: readonly_array(identity, "exo_" + key + "_counts")
             for key in ("person", "static", "rigid")}, valid=valid)
        if not np.array_equal(available > 0, evidence["exported_position_available"]):
            raise ValueError("Saved EXO availability differs from original first-hit count evidence")
        if not np.allclose(confidence, evidence["exported_position_confidence"], atol=1e-6):
            raise ValueError("Saved EXO confidence differs from original first-hit count evidence")
        observation[..., :3] = np.where(available[..., None] > 0, trajectory / 2, 0)
        metadata[..., 0], metadata[..., 2] = available, confidence
        occluded = evidence["has_occluding_surface"]
        note = dict(kind="saved_synthetic_external_joint_observations",
            availability_rule=evidence["availability_rule"],
            occlusion_evidence=evidence["note"], real_tracker=False)
    return dict(observations=observation, observation_meta=metadata, source_occluded=occluded), note


def _observer_layers(identity: dict, n: int, joints: np.ndarray) -> tuple[dict, dict]:
    camera, camera_meta = camera_layers(identity, n, frustum_depth_m=.8)
    current, current_meta = instant_scene_layers(identity, n)
    # The union is explicitly an offline recording memory, including later
    # observed frames. It is not the full mesh and is not a causal map.
    static = np.concatenate([current[f"current_static_{i}"] for i in range(n)])
    if len(static):
        _, unique = np.unique(np.floor(static / .025).astype(np.int64), axis=0, return_index=True)
        static = static[np.sort(unique)]
        if len(static) > 50000:
            static = static[np.linspace(0, len(static) - 1, 50000, dtype=int)]
    observations, obs_meta = _observations(identity, n, joints)
    return dict(**camera, **current, **observations, visible_static=static), dict(
        camera=camera_meta, current_scene=current_meta, observation=obs_meta,
        visible_union=dict(points=len(static), voxel_m=.025, causal=False,
            source="saved 512-sample first-hit static observations across this complete sequence",
            note="Includes future frames in this recording; full source mesh is display-only"))


def prepare(manifest: Path, output: Path) -> Path:
    import torch

    torch.set_num_threads(1)
    rows = _source_rows(manifest.resolve())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = body_model_factory("g1")
    assets = g1_template_assets()
    faces = np.asarray(model.faces, np.int32)
    implementation = {str(path.relative_to(REPO)): file_sha256(path) for path in
        (Path(__file__), Path(sys.modules[MotionViewer.__module__].__file__),
         Path(sys.modules[camera_layers.__module__].__file__),
         REPO / "experiments/offline_camera_retrain_v1/ops/grail_source.py")}
    cases = []
    for row in rows:
        native = row["native_body"]
        if native.get("template") != G1_TEMPLATE or native.get("template_sha256") != assets["sha256"]:
            raise ValueError("Source bundle does not bind the pinned G1 source template")
        if native.get("scale") != 1 or np.any(np.asarray(native.get("betas")) != 0):
            raise ValueError("Unexpected per-motion scaling or shape for pinned G1 source template")
        n = int(row["frames"])
        params = {name[len("source_pose_"):-4]: np.asarray(readonly_array(row, name[:-4]))
                  for name in row["output_sha256"] if name.startswith("source_pose_") and name.endswith(".npy")}
        if not params or any(len(a) != n for a in params.values()):
            raise ValueError("Source SMPL-X pose arrays do not span the complete sequence")
        rotation = readonly_array(row, "source_world_rotation")
        offset = readonly_array(row, "source_world_offset")
        vchunks, jchunks = [], []
        for lo in range(0, n, 16):
            values = {key: torch.from_numpy(value[lo:min(n, lo + 16)].copy())
                      for key, value in params.items()}
            with torch.inference_mode():
                forward = model(**values, return_verts=True)
            # Identical world transform to forward_chunk; the pinned G1
            # template has baked scale=1. Avoid importing legacy corpus tools.
            v = (forward.vertices.numpy() @ rotation.T + offset).astype(np.float32)
            j = (forward.joints[:, :22].numpy() @ rotation.T + offset).astype(np.float32)
            vchunks.append(v)
            jchunks.append(j)
        vertices, joints = np.concatenate(vchunks), np.concatenate(jchunks)
        expected = np.asarray(readonly_array(row, "motion_world"))[:, -66:].reshape(n, 22, 3) * 2
        error = float(np.max(np.abs(joints - expected)))
        if error > 1e-5:
            raise ValueError(f"Source SMPL-X mesh joints differ from exported motion: {error}")
        arrays = dict(vertices=vertices, joints=joints, faces=faces,
            rest=np.asarray(readonly_array(row, "rest")), **_full_scene(row))
        center = (joints.reshape(-1, 3).min(0) + joints.reshape(-1, 3).max(0)) / 2
        arrays.update(center=center.astype(np.float32),
            radius=np.asarray(max(1., np.linalg.norm(joints - center, axis=-1).max()), np.float32))
        meta = dict(data_only=True, warning=WARNING, source_sequence_id=row["sequence_id"],
            data_frames=n, fps=30, duration_s=(n - 1) / 30, full_sequence=True, model_inference=False,
            source_body_template=G1_TEMPLATE, fk_joint_max_error_m=error,
            source_identity=row, implementation_sha256=implementation, observers={})
        for mode in ("ego", "exo"):
            identity = dict(row, mode=mode, start=0, origin=[0., 0., 0.], anchor=np.eye(3).tolist())
            layer, evidence = _observer_layers(identity, n, joints)
            arrays.update({mode + "__" + name: value for name, value in layer.items()})
            meta["observers"][mode] = dict(identity=identity, display_geometry=evidence,
                scene_errors=[], data_only=True, generation=dict(recorded_guidance="未运行模型；无物理修正"))
        arrays["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False, allow_nan=False))
        digest = _json_hash(dict(source_metadata_sha256=row["metadata_sha256"], implementation=implementation))
        cache = output / "cache" / (digest + ".npz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(cache)
        cache_hash = file_sha256(cache)
        record_path = cache.with_suffix(".json")
        atomic_json(record_path, dict(schema=SCHEMA, data_only=True, warning=WARNING,
            cache=str(cache), cache_sha256=cache_hash, source_identity=row,
            data_frames=n, full_sequence=True, implementation_sha256=implementation,
            fk_joint_max_error_m=error))
        cases.append(dict(label=row["sequence_id"], dataset="grail", category=row["category"],
            sequence_id=row["sequence_id"], data_frames=n, data_only=True,
            selection_reason="完整原始重建序列；只检视源动作、合成相机与场景观测，未运行生成模型。",
            models=[dict(key="grail_source", label=SOURCE_LABEL, mode=mode,
                case=dict(cache=str(cache), cache_sha256=cache_hash, cache_manifest=str(record_path)))
                for mode in ("ego", "exo")]))
        print(json.dumps(dict(prepared=row["sequence_id"], data_frames=n,
            fk_joint_max_error_m=error, cache_bytes=cache.stat().st_size), ensure_ascii=False), flush=True)
    published = output / "PREPARED_CASES.json"
    atomic_json(published, dict(schema=SCHEMA, data_only=True, warning=WARNING,
        source_manifest=str(manifest.resolve()), source_manifest_sha256=file_sha256(manifest), cases=cases))
    return published


def validate_prepared(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("data_only") is not True:
        raise ValueError("Expected a GRAIL source-only preview manifest")
    for case in manifest["cases"]:
        if case.get("data_only") is not True or {m["key"] for m in case["models"]} != {"grail_source"}:
            raise ValueError("Source preview cannot contain generated model layers")
        cache_paths = {entry["case"]["cache"] for entry in case["models"]}
        if len(cache_paths) != 1 or {entry["mode"] for entry in case["models"]} != {"ego", "exo"}:
            raise ValueError("EGO/EXO must share one complete source motion")
        record = case["models"][0]["case"]
        if file_sha256(Path(record["cache"])) != record["cache_sha256"]:
            raise ValueError("Prepared source playback cache SHA256 mismatch")
        metadata = json.loads(Path(record["cache_manifest"]).read_text())
        source = metadata["source_identity"]
        if file_sha256(Path(source["path"]) / "metadata.json") != source["metadata_sha256"]:
            raise ValueError("Exported bundle changed after preview preparation")
        if metadata.get("data_frames") != case["data_frames"] or not metadata.get("full_sequence"):
            raise ValueError("Source preview length identity mismatch")
    return manifest["cases"]


@contextmanager
def _source_ui_labels():
    """Scope base UI wording changes to construction of this source-only view.

    MotionViewer owns the playback controls and scene layers. These wrappers
    substitute its evaluation-specific text without editing its sealed source.
    """
    import viser
    api = viser.GuiApi
    originals = {name: getattr(api, name) for name in ("add_markdown", "add_checkbox", "add_slider")}

    def markdown(self, content, *args, **kwargs):
        if content.startswith("## 共享生成器"):
            content = "## " + WARNING + "\n\n紫色人体是发布数据的原始重建。EGO / EXO 为同一完整动作的两种合成观测。"
        elif content.startswith("输入参考人体是"):
            content = ("此页未运行扩散模型、适配层或 tracker，不显示生成指标或训练步数。"
                "源重建仍可能含接触和穿插误差。红点仅表示 EXO 原始 first-hit 遮挡证据，部分遮挡可与非零输入并存。\n\n"
                "完整场景使用导出保存的对象及地板；若导出隔离了地板，此页不另加地板。"
                "可见静态并集来自本序列所有帧，包含后续观测；当前帧只显示保存的512点环境采样。"
                "相机来自固定头部安装或初始根位置外部观察者，不按未来人体动作拟合。")
        return originals["add_markdown"](self, content, *args, **kwargs)

    def checkbox(self, label, initial_value, *args, **kwargs):
        if label == "GT 人体（蓝）":
            initial_value = False
        label = {"跟随 GT 人体": "跟随原始重建人体", "完整静态场景（GT参考）": "完整源场景（仅展示）",
            "本录制可见静态并集（实际输入）": "本序列可见静态并集（含后续帧）",
            "实际输入关节点 + 置信颜色": "导出的观测关节点 + 置信颜色"}.get(label, label)
        return originals["add_checkbox"](self, label, initial_value, *args, **kwargs)

    def slider(self, label, *args, **kwargs):
        label = {"帧 / 30 Hz": "数据帧 / 30 Hz", "预测人体不透明度": "原始重建人体不透明度"}.get(label, label)
        return originals["add_slider"](self, label, *args, **kwargs)

    api.add_markdown, api.add_checkbox, api.add_slider = markdown, checkbox, slider
    try:
        yield
    finally:
        for name, method in originals.items():
            setattr(api, name, method)


class GrailSourceViewer(MotionViewer):
    """The existing motion viewer with honest source-only labels and cache."""
    COLORS = dict(MotionViewer.COLORS, grail_source=(185, 40, 160))

    def __init__(self, cases: list[dict], port: int):
        with _source_ui_labels():
            super().__init__(cases, port)

    def read_case(self, index: int) -> list[dict]:
        if index not in self.cache:
            case = self.cases[index]
            with np.load(case["models"][0]["case"]["cache"], allow_pickle=False) as saved:
                shared = {key: np.asarray(saved[key]) for key in saved.files}
            metadata = json.loads(str(shared["metadata_json"]))
            if not metadata.get("data_only") or not metadata.get("full_sequence"):
                raise ValueError("This viewer requires a complete source-only sequence")
            common = {k: v for k, v in shared.items() if "__" not in k and k != "metadata_json"}
            # Compatibility aliases are references to one source reconstruction,
            # not stored predictions or independent ground-truth comparisons.
            for regime in ("clean", "noisy"):
                common[regime + "_vertices"] = common[regime + "_reference_vertices"] = shared["vertices"]
                common[regime + "_joints"] = common[regime + "_reference_joints"] = shared["joints"]
            common["truth_vertices"], common["truth_joints"] = shared["vertices"], shared["joints"]
            result = []
            for entry in case["models"]:
                mode = entry["mode"]
                data = dict(common)
                data.update({key[len(mode) + 2:]: value for key, value in shared.items()
                    if key.startswith(mode + "__")})
                for regime in ("clean", "noisy"):
                    data[regime + "_observations"] = data["observations"]
                    data[regime + "_observation_meta"] = data["observation_meta"]
                result.append(dict(entry=entry, data=data, meta=metadata["observers"][mode]))
            self.cache[index] = result
        self.cache.move_to_end(index)
        while len(self.cache) > 1:
            self.cache.popitem(last=False)
        return self.cache[index]

    def load_case(self) -> None:
        with self.lock, self.server.atomic():
            super().load_case()
            self.regime.visible = False
            self.regime.disabled = True
            for panel in self.panels.values():
                for key in ("gt", "reference", "uncertainty", "reference_source"):
                    panel[key].visible = False
                    panel[key].disabled = True
                panel["gt"].value = panel["reference"].value = panel["uncertainty"].value = False
            for button in self.preset_buttons:
                button.visible = False
                button.disabled = True
            self.update()

    def update(self) -> None:
        if self.loading:
            return
        with self.lock, self.server.atomic():
            super().update()
            t = int(self.frame.value)
            case = self.cases[self.index]
            n = int(case["data_frames"])
            self.info.content = (f"**{WARNING}**\n\n"
                f"数据帧 {t + 1}/{n} · {t / 30:.2f}/{(n - 1) / 30:.2f} 秒 · 完整序列 · 30 Hz\n\n"
                "此页没有模型预测，未计算预测误差；帧数不是训练步数。")
            for mode, runtime in self.runtime.items():
                panel, data = self.panels[mode], runtime["data"]
                panel["status"].content = ("**" + mode.upper() + " 合成观测** · 同一原始重建人体；未运行生成模型。")
                evidence = runtime["meta"]["display_geometry"]
                active = int((data["observation_meta"][t, :, 0] > 0).sum())
                count = int(data["source_occluded"][t].sum())
                visibility_note = (f"有原始 first-hit 遮挡证据 {count} 个部位。" if mode == "exo" else
                    "EGO 仅提供头部位置 / 旋转，不以缺失关节推断全身遮挡。")
                panel["info"].content = (f"本帧提供 {active}/22 个关节位置；{visibility_note}\n\n"
                    f"相机：{evidence['camera']['source_kind']}；视锥显示长度 {evidence['camera']['frustum_depth_m']:.2f} m"
                    " 仅为示意大小，不是可见距离，也不改变真实 FOV。\n\n"
                    + ("固定头部安装，随当前源头姿态移动。" if mode == "ego" else
                       "外部相机仅按初始根位置选定，此后固定；观测由可见人体表面构造，尚非真实 tracker。")
                    + "\n\n可见静态并集含本序列后续帧，完整源场景只作检查；未执行贴地或物理修正。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="CPU prepare complete exported source sequences")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve", help="Serve already prepared source data with the shared Viser viewer")
    serve.add_argument("--manifest", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8840)
    args = parser.parse_args()
    if args.command == "prepare":
        print(str(prepare(args.manifest, args.output)), flush=True)
        return
    cases = validate_prepared(args.manifest.resolve())
    viewer = GrailSourceViewer(cases, args.port)
    atomic_json(args.manifest.resolve().parent / "SERVER.json", dict(pid=os.getpid(),
        url=f"http://localhost:{args.port}/", bind="127.0.0.1", manifest=str(args.manifest.resolve()),
        data_only=True, warning=WARNING, model_inference=False))
    viewer.run()


if __name__ == "__main__":
    main()
