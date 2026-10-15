"""Interrupted-review recovery without editing the six sealed extensions.

Only this recovery process replaces the rejected-checkpoint cleanup function.
Already compacted models require matching model-hash cleanup evidence. A new
write-ahead journal also covers interruptions during future recovery cleanup.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
JOURNAL = "RESUME_STORAGE_CLEANUP.json"


def model_hash(state):
    model = state.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("Checkpoint has no model tensors")
    digest = hashlib.sha256()
    for name, value in model.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("Invalid model entry")
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _records(path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    records = data.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Invalid cleanup evidence: " + str(path))
    return records


def compact_rejected_safely(manager, selected):
    """Idempotent cleanup; no absent optimizer is accepted without evidence."""
    root = Path(manager.ROOT)
    if selected not in manager.NAMES:
        raise ValueError("Unknown selected arm")
    journal_path = root / JOURNAL
    original = _records(root / "STORAGE_CLEANUP.json")
    recovery = _records(journal_path)
    by_arm = {row["arm"]: row for row in recovery if "arm" in row}

    def persist(status):
        manager.write_json(journal_path, dict(
            status=status, updated_at=manager.now(), records=list(by_arm.values()),
            preserved_resumable=["original parent", "control", selected],
            scope="Recovery-only write-ahead model-hash cleanup evidence"))

    for name in manager.NAMES:
        if name in (selected, "control"):
            continue
        checkpoint = root / "arms" / name / "training" / "last.pt"
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        digest = model_hash(state)
        if "optimizer" not in state:
            if state.get("evaluation_only") is not True:
                raise ValueError("Missing optimizer without evaluation-only declaration: " + name)
            evidence = [row for row in original + list(by_arm.values())
                        if row.get("arm") == name and row.get("model_sha256") == digest
                        and (row.get("model_verified_unchanged") is True
                             or (row.get("status") == "prepared"
                                 and row.get("model_verified_before_cleanup") is True))]
            if not evidence:
                raise ValueError("Evaluation-only model lacks matching cleanup hash evidence: " + name)
            by_arm[name] = dict(evidence[-1], status="completed", model_verified_unchanged=True,
                                recovery_verified_at=manager.now(), already_compacted=True,
                                after_bytes=checkpoint.stat().st_size)
            del state
            persist("running")
            continue
        record = dict(arm=name, status="prepared", model_sha256=digest,
                      model_verified_before_cleanup=True, prepared_at=manager.now(),
                      before_bytes=checkpoint.stat().st_size, already_compacted=False)
        by_arm[name] = record
        persist("running")  # Durable evidence precedes replacement of the checkpoint.
        state.pop("optimizer")
        state["evaluation_only"] = True
        state["optimizer_removed_reason"] = "Finished rejected candidate; no resume planned. Model and all small RNG/sampler states retained."
        manager.save_checkpoint(state, checkpoint)
        del state
        kept = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        if ("optimizer" in kept or kept.get("evaluation_only") is not True
                or model_hash(kept) != digest):
            raise ValueError("Compacted checkpoint verification failed: " + name)
        del kept
        record.update(status="completed", model_verified_unchanged=True,
                      after_bytes=checkpoint.stat().st_size, completed_at=manager.now())
        persist("running")
    persist("completed")


def assert_no_live_review(root=ROOT, *, proc_root=Path("/proc")):
    """Reject matching live owner PIDs; unrelated reused PIDs are ignored."""
    root = Path(root)
    status_path = root / "STATUS.json"
    if not status_path.exists():
        raise ValueError("Missing STATUS.json; this entry only resumes an existing review")
    status = json.loads(status_path.read_text())
    for field in ("pid", "worker_pid", "fallback_baseline_pid"):
        pid = status.get(field)
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            continue
        try:
            args = (Path(proc_root) / str(pid) / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        scripts = {str(root / name).encode() for name in ("run_review.py", "resume_review.py", "worker.py")}
        scripts.add(str(root.parent / "native30_gait_v2_ego55k_oct15" / "run_training.py").encode())
        if scripts.intersection(args):
            raise RuntimeError(f"Review process still active: {field}={pid}; do not start a concurrent manager")


def main():
    assert_no_live_review()
    import run_review as manager
    manager.compact_rejected = lambda selected: compact_rejected_safely(manager, selected)
    if "--resume-manager" not in sys.argv:
        sys.argv.append("--resume-manager")
    manager.main()


if __name__ == "__main__":
    main()
