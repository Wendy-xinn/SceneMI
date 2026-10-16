"""Keep one evaluation-only best checkpoint without changing the running trainer.

Selection is deliberately provisional: one preregistered fixed RICH window,
scene ON, arithmetic mean of clean/noisy raw world MPJPE. Rotating windows and
TEST never enter selection. Historical metrics do not imply weights still exist.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


HERE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = HERE / "runs/native30_gait_v2_ego55k_oct15"
SELECTION_STATUS = "provisional_fixed_rich_window"


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def tensor_digest(model):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.items()):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError("Nonfinite model tensor: " + name)
        metadata = dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape))
        h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        h.update(memoryview(raw).cast("B"))
    return h.hexdigest()


def fixed_rich_score(record):
    """Reject ambiguity and return the unweighted two-regime mean and identity."""
    rows = [r for r in record["cases"]
            if r.get("dataset") == "rich" and r.get("panel") == "fixed"
            and r.get("mode") == "ego"
            and r.get("scene_variant", "scene_on") == "scene_on"]
    if len(rows) != 2 or {r["regime"] for r in rows} != {"clean", "noisy"}:
        raise ValueError("Fixed RICH clean/noisy pair is incomplete or ambiguous")
    identity_fields = ("dataset", "recording", "group", "mode", "sequence_id",
                       "source_first", "start", "case_seed", "joints", "panel")
    identities = [{k: r.get(k) for k in identity_fields} for r in rows]
    if identities[0] != identities[1] or any(r["step"] != record["step"] for r in rows):
        raise ValueError("Fixed RICH pair identity/step mismatch")
    values = {}
    for row in rows:
        metric = row["metrics"]["mpjpe_mm"]
        if metric["count"] <= 0 or not math.isfinite(metric["value"]):
            raise ValueError("Invalid fixed RICH MPJPE")
        values[row["regime"]] = float(metric["value"])
    return dict(step=int(record["step"]), score_mm=sum(values.values()) / 2,
                regime_mpjpe_mm=values, identity=identities[0],
                validation_record_sha256=hashlib.sha256(json.dumps(
                    record, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def validation_scores(path):
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()  # The trainer may currently be publishing its final line.
    scores = {}
    identity = None
    for line in lines:
        score = fixed_rich_score(json.loads(line))
        if identity is None:
            identity = score["identity"]
        if score["identity"] != identity:
            raise ValueError("The preregistered fixed RICH window changed")
        old = scores.get(score["step"])
        if old is not None and old != score:
            raise ValueError("Conflicting validation records for one checkpoint step")
        scores[score["step"]] = score
    return sorted(scores.values(), key=lambda item: item["step"])


def verify_sources(run):
    effective_path = run / "prepared/effective_config.json"
    protocol = json.loads((run / "continuous_training_oct16/PROTOCOL.json").read_text())
    effective = json.loads(effective_path.read_text())
    expected = {**effective["code_sha256"], **protocol["expected_source_sha256"]}
    for path, expected_sha in expected.items():
        if sha(path) != expected_sha:
            raise ValueError("Sealed training dependency changed: " + path)
    for path, key in ((run / "config.json", "original_config_sha256"),
                      (effective_path, "original_effective_config_sha256")):
        if sha(path) != protocol[key]:
            raise ValueError("Sealed training configuration changed: " + str(path))
    return effective, dict(verified_at=now(), sealed_sources=len(expected),
                          config_sha256=sha(run / "config.json"),
                          effective_config_sha256=sha(effective_path))


def recover_best(training):
    import torch
    target = training / "best.pt"
    if not target.exists():
        if (training / "best.json").exists():
            raise ValueError("best.json exists without its model")
        return None
    state = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
    embedded = state["checkpoint_selection"]
    if embedded["selection_status"] != SELECTION_STATUS:
        raise ValueError("Existing best checkpoint uses a different selection protocol")
    digest = tensor_digest(state["model"])
    if digest != embedded["source_model_tensor_sha256"]:
        raise ValueError("Existing best checkpoint model digest mismatch")
    metadata = dict(embedded, checkpoint_sha256=sha(target), checkpoint_bytes=target.stat().st_size,
                    compact_model_tensor_sha256=digest, tensor_equality_verified=True)
    write_json(training / "best.json", metadata)
    del state
    gc.collect()
    return metadata


def capture_if_step(run, score, historical_best, source_override=None, source_provenance=None):
    """Pin one immutable trainer inode, then require its exact evaluated step."""
    import torch
    training = run / "training"
    source = Path(source_override) if source_override else training / "last.pt"
    pinned = training / f".guardian-pinned-{os.getpid()}.pt"
    temporary = training / f".guardian-best-{os.getpid()}.tmp"
    if pinned.exists() or temporary.exists():
        raise ValueError("Unresolved guardian temporary file")
    os.link(source, pinned)
    try:
        state = torch.load(pinned, map_location="cpu", mmap=True, weights_only=False)
        step = int(state["step"])
        if step != score["step"]:
            return dict(status="waiting" if step < score["step"] else "weights_unavailable",
                        published_step=step)
        effective, audit = verify_sources(run)
        if state["effective_config"] != effective or state["planned_steps"] != 55000:
            raise ValueError("Published model/config does not match protected baseline")
        if shutil.disk_usage(training).free < 2 * 1024**3:
            raise OSError("Less than 2 GiB free for safe best-checkpoint replacement")
        source_digest = tensor_digest(state["model"])
        metadata = dict(
            saved_at=now(), selection_status=SELECTION_STATUS,
            selection_metric="arithmetic mean of clean/noisy raw world MPJPE, one fixed RICH VAL window, scene ON, head-only ego",
            selection_scope="Best among exactly matched checkpoints captured by this guardian; not full-RICH or historical global best",
            checkpoint_step=step, score=score,
            historical_best_metric=historical_best,
            source_checkpoint=str(source), source_checkpoint_sha256=sha(pinned),
            source_provenance=source_provenance,
            source_model_tensor_sha256=source_digest, source_inode=pinned.stat().st_ino,
            guardian_source=str(Path(__file__).resolve()), guardian_sha256=sha(Path(__file__)),
            source_audit=audit, evaluation_only=True, optimizer_and_training_RNG_untouched=True,
            optimizer_removed_from_copy=True, storage_policy="One best.pt plus original resumable last.pt; best is model-only")
        compact = dict(step=step, model=state["model"], effective_config=effective,
                       planned_steps=state["planned_steps"], evaluation_only=True,
                       checkpoint_selection=metadata)
        with temporary.open("wb") as stream:
            torch.save(compact, stream)
            stream.flush()
            os.fsync(stream.fileno())
        check = torch.load(temporary, map_location="cpu", mmap=True, weights_only=False)
        compact_digest = tensor_digest(check["model"])
        if compact_digest != source_digest or set(check["model"]) != set(state["model"]):
            raise ValueError("Saved best model differs from exact trainer checkpoint")
        # Both independent digests and direct tensor equality are recorded.
        if not all(torch.equal(tensor, check["model"][name])
                   for name, tensor in state["model"].items()):
            raise ValueError("Direct model tensor equality failed")
        verify_sources(run)
        target = training / "best.pt"
        metadata.update(compact_model_tensor_sha256=compact_digest,
                        tensor_equality_verified=True, checkpoint_sha256=sha(temporary),
                        checkpoint_bytes=temporary.stat().st_size)
        os.replace(temporary, target)
        sync_directory(training)
        write_json(training / "best.json", metadata)
        del check, compact, state
        gc.collect()
        return dict(status="saved", best=metadata)
    finally:
        pinned.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def seed_from_evaluation(run, folder):
    """Use an already evaluated, exactly matched snapshot without new inference."""
    import numpy as np
    import torch
    import sys
    import io
    repo = HERE.parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    effective, _ = verify_sources(run)
    scores = validation_scores(run / "training/validation.jsonl")
    expected_identity = scores[-1]["identity"]
    plan = json.loads((folder / "PLAN.json").read_text())
    protocol = json.loads((folder / "PROTOCOL.json").read_text())
    frozen = json.loads((folder / "FROZEN_CHECKPOINT.json").read_text())
    if (protocol["scene"], protocol["mode"], protocol["frames"], protocol["fps"], protocol["ddim_steps"]) != ("on", "ego", 192, 30, 20):
        raise ValueError("Seed evaluation sampling/input protocol mismatch")
    if protocol["effective_config_sha256"] != sha(run / "prepared/effective_config.json"):
        raise ValueError("Seed evaluation configuration mismatch")
    if protocol["plan_sha256"] != sha(folder / "PLAN.json"):
        raise ValueError("Seed evaluation plan hash mismatch")
    matching = [item for item in plan["plan"] if all(item.get(key) == value
                for key, value in expected_identity.items() if key not in ("panel",))]
    if len(matching) != 1:
        raise ValueError("Seed evaluation does not contain exactly one matching fixed window")
    item = matching[0]
    # Inference inputs and GT are independent of model weights. Compare them
    # byte-for-byte with the actual trainer monitor archive, even if its step
    # differs from the frozen model being evaluated here.
    monitor_bytes = (run / "training/validation_latest.npz").read_bytes()
    monitor = np.load(io.BytesIO(monitor_bytes), allow_pickle=False)
    monitor_step = int(monitor["step"])
    monitor_records = [json.loads(line) for line in (run / "training/validation.jsonl").read_text().splitlines()]
    monitor_records = [record for record in monitor_records if record["step"] == monitor_step]
    if len(monitor_records) != 1:
        raise ValueError("The monitor input archive lacks one exact validation record")
    monitor_case_rows = [row for row in monitor_records[0]["cases"]
                         if row.get("dataset") == "rich" and row.get("panel") == "fixed"
                         and row.get("mode") == "ego"
                         and row.get("scene_variant", "scene_on") == "scene_on"]
    if len(monitor_case_rows) != 2 or len({row["case"] for row in monitor_case_rows}) != 1:
        raise ValueError("The monitor fixed RICH case is ambiguous")
    monitor_case = monitor_case_rows[0]["case"]
    lines = (folder / "rows.jsonl").read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    rows = [json.loads(line) for line in lines]
    selected = [row for row in rows if row["case"] == item["case"]]
    if len(selected) != 2 or {row["regime"] for row in selected} != {"clean", "noisy"}:
        raise ValueError("Seed fixed clean/noisy evaluation pair is incomplete")
    model_path = Path(frozen["evaluation_checkpoint"])
    if (model_path != (folder / "checkpoint_eval.pt").resolve()
            or sha(model_path) != frozen["evaluation_checkpoint_sha256"]
            or protocol["checkpoint_sha256"] != frozen["evaluation_checkpoint_sha256"]):
        raise ValueError("Seed compact checkpoint SHA/path mismatch")
    state = torch.load(model_path, map_location="cpu", mmap=True, weights_only=False)
    digest = tensor_digest(state["model"])
    if (state["effective_config"] != effective
            or digest != frozen["source_model_tensor_sha256"]
            or digest != frozen["compact_model_tensor_sha256"]
            or not frozen["direct_source_tensor_digest_comparison"]):
        raise ValueError("Seed/source model tensor proof mismatch")
    step = state["step"]
    del state
    adapted = []
    artifact_proofs = []
    for row in selected:
        if row["checkpoint_step"] != step or frozen["checkpoint_step"] != step or protocol["checkpoint_step"] != step:
            raise ValueError("Seed evaluation/checkpoint step mismatch")
        for key, value in expected_identity.items():
            if key not in ("panel", "mode", "joints") and row.get(key) != value:
                raise ValueError("Seed prediction identity mismatch: " + key)
        path = (folder / row["motion_file"]).resolve()
        if folder.resolve() not in path.parents or sha(path) != row["motion_sha256"]:
            raise ValueError("Seed prediction artifact path/SHA mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if int(arrays["step"]) != step:
                raise ValueError("Seed saved motion checkpoint step mismatch")
            for seed_key, monitor_key in (("truth", f"{monitor_case}_truth"),
                                          ("rest", f"{monitor_case}_rest"),
                                          ("observations", f"{monitor_case}_{row['regime']}_observations"),
                                          ("observation_meta", f"{monitor_case}_{row['regime']}_observation_meta"),
                                          ("reference", f"{monitor_case}_{row['regime']}_reference_motion")):
                if not np.array_equal(arrays[seed_key], monitor[monitor_key]):
                    raise ValueError("Seed and trainer monitor inputs/GT differ: " + seed_key)
            available_joints = np.flatnonzero(np.any(arrays["observation_meta"][..., 0] > 0, axis=0)).tolist()
            if available_joints != [15]:
                raise ValueError("Seed evaluation is not head-only ego")
            prediction = torch.from_numpy(arrays["prediction"])[None]
            truth = torch.from_numpy(arrays["truth"])[None]
            rest = torch.from_numpy(arrays["rest"])[None]
            error = ((forward_kinematics(prediction, rest) - forward_kinematics(truth, rest))
                     .norm(dim=-1).mean() * 1000).item()
            if abs(error - row["metrics"]["mpjpe_mm"]["value"]) > 0.001:
                raise ValueError("Seed MPJPE failed independent CPU recomputation")
        adapted.append(dict(row, step=step, mode="ego", panel="fixed", joints=[15]))
        artifact_proofs.append(dict(path=str(path), sha256=row["motion_sha256"],
                                    regime=row["regime"], cpu_recomputed_mpjpe_mm=error))
    score = fixed_rich_score(dict(step=step, cases=adapted))
    proof = dict(type="existing_frozen_evaluation_exact_fixed_window",
                 evaluation_folder=str(folder), metadata_sha256=sha(folder / "FROZEN_CHECKPOINT.json"),
                 original_trainer_checkpoint_sha256=frozen["source_checkpoint_sha256"],
                 original_trainer_model_tensor_sha256=frozen["source_model_tensor_sha256"],
                 plan_sha256=sha(folder / "PLAN.json"), protocol_sha256=sha(folder / "PROTOCOL.json"),
                 artifact_proofs=artifact_proofs, metrics_independently_recomputed_on_CPU=True,
                 monitor_input_archive_sha256=hashlib.sha256(monitor_bytes).hexdigest(),
                 monitor_input_archive_step=monitor_step,
                 trainer_monitor_inputs_GT_and_reference_exactly_equal=True,
                 input_comparison_does_not_pair_monitor_step_metrics_with_seed_weights=True)
    monitor.close()
    return score, model_path, proof, min(scores, key=lambda item: item["score_mm"])


def run_guardian(run, poll_seconds=2.0, once=False, seed_evaluation=None):
    import torch
    torch.set_num_threads(1)
    training = run / "training"
    lock_stream = (training / ".best_guardian.lock").open("a")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(run)
    best = recover_best(training)
    if seed_evaluation is not None:
        score, source, proof, historical = seed_from_evaluation(run, seed_evaluation.resolve())
        if best is None or score["score_mm"] < best["score"]["score_mm"]:
            result = capture_if_step(run, score, historical, source_override=source,
                                     source_provenance=proof)
            if result["status"] != "saved":
                raise ValueError("The exact seeded evaluation checkpoint could not be saved")
            best = result["best"]
            print(json.dumps(dict(event="best_seeded_from_verified_evaluation", step=score["step"],
                                  score_mm=score["score_mm"])), flush=True)
    handled = set()
    started = now()
    previous_inode = None
    while True:
        scores = validation_scores(training / "validation.jsonl")
        historical_best = min(scores, key=lambda item: item["score_mm"]) if scores else None
        source_stat = (training / "last.pt").stat()
        inode = (source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)
        if scores and inode != previous_inode:
            # Only current/near-future validation can still have matching weights.
            # Published step is read from the pinned file, never inferred from time.
            score = scores[-1]
            if score["step"] not in handled:
                if best is not None and score["score_mm"] >= best["score"]["score_mm"]:
                    handled.add(score["step"])
                else:
                    result = capture_if_step(run, score, historical_best)
                    if result["status"] == "saved":
                        best = result["best"]
                        handled.add(score["step"])
                        print(json.dumps(dict(event="best_saved", step=score["step"],
                                              score_mm=score["score_mm"])), flush=True)
                    elif result["status"] == "weights_unavailable":
                        handled.add(score["step"])
                        print(json.dumps(dict(event="evaluated_weights_already_overwritten",
                                              step=score["step"], published_step=result["published_step"])), flush=True)
            previous_inode = inode
        status = dict(status="watching", pid=os.getpid(), started_at=started, checked_at=now(),
                      poll_seconds=poll_seconds, selection_status=SELECTION_STATUS,
                      best_step=best["checkpoint_step"] if best else None,
                      best_score_mm=best["score"]["score_mm"] if best else None,
                      latest_validation_step=scores[-1]["step"] if scores else None,
                      historical_best_metric=historical_best,
                      historical_weights_cannot_be_reconstructed_from_metrics=True,
                      protected_evaluation_snapshots=[12600, 15100],
                      historical_12000_weights="not retained; metrics remain, cannot claim saved historical optimum",
                      disk_free_bytes=shutil.disk_usage(training).free,
                      trainer_and_resumable_last_untouched=True,
                      guardian_sha256=sha(Path(__file__)))
        write_json(training / "BEST_GUARDIAN_STATUS.json", status)
        if once:
            return status
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--seed-evaluation", type=Path,
                        help="Existing frozen RICH evaluation folder, validated against the fixed monitor window")
    args = parser.parse_args()
    if args.poll_seconds < 0.2:
        parser.error("Polling faster than 0.2 seconds is unnecessary")
    try:
        run_guardian(args.run.resolve(), args.poll_seconds, args.once, args.seed_evaluation)
    except Exception as exc:
        write_json(args.run / "training/BEST_GUARDIAN_STATUS.json",
                   dict(status="failed", pid=os.getpid(), checked_at=now(), error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
