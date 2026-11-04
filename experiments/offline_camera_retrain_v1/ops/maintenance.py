"""Inventory and remove explicitly retired artifacts, retaining verifiable records.

Policies enumerate source files and retired output directories. Dataset directories,
symlinks, and protected files cannot be removed. Planning never deletes anything.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/offline_camera_retrain_v1"
ARTIFACT_SUFFIXES = {".pt", ".pth", ".npy", ".npz", ".mp4", ".webm", ".sqlite3"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def local_file(value, *, root=None):
    root = ROOT if root is None else root
    path = root / value
    if path.absolute() != path.resolve() or not path.is_relative_to(root):
        raise ValueError(f"Symlink or external path is not a cleanup target: {path}")
    if not path.is_file():
        raise ValueError(f"Missing cleanup input: {path}")
    return path


def snapshot_sources(destination):
    """Keep one small, exact recovery bundle before retiring old entry points."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Refuse to overwrite a source recovery bundle")
    paths = sorted({*BASE.glob("*.py"), *(BASE / "ops").glob("*.py"),
                    *(BASE / "configs").glob("*.json")})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(destination) as archive:
        for path in paths:
            member = archive.extractfile(str(path.relative_to(ROOT)))
            if member is None or hashlib.sha256(member.read()).hexdigest() != digest(path):
                raise ValueError(f"Recovery verification failed: {path}")
    return {"path": str(destination.relative_to(ROOT)), "sha256": digest(destination),
            "files": len(paths), "bytes": destination.stat().st_size}


def plan(policy_path, output):
    policy_path = Path(policy_path)
    policy = json.loads(policy_path.read_text())
    protected = {str((ROOT / p).resolve()): sha for p, sha in policy["protected"].items()}
    protected_directories = [(ROOT / p).resolve() for p in policy.get("protected_directories", [])]
    candidates = {}
    for value in policy["retired_sources"]:
        path = local_file(value)
        if path.parent != BASE or path.suffix != ".py":
            raise ValueError("Source retirement must name experiment-root Python files")
        candidates[path] = "retired_entry_or_test; exact source recovery bundle retained"
    explicit_artifacts = policy.get("retired_artifacts", [])
    for value in explicit_artifacts:
        path = local_file(value)
        if not path.is_relative_to(ROOT / "runs") or path.suffix not in ARTIFACT_SUFFIXES:
            raise ValueError("Explicit artifact retirement must name a generated file in root runs")
        candidates[path] = "unselected_or_rejected_artifact; configuration and evaluation evidence retained"
    output_roots = []
    for value in policy["retired_output_roots"]:
        path = ROOT / value
        if path.absolute() != path.resolve() or not path.is_dir():
            raise ValueError(f"Output root must be a real local directory: {path}")
        if not any(path.is_relative_to(BASE / folder) for folder in ("runs", "archive")):
            raise ValueError("Cleanup is restricted to experiment outputs")
        output_roots.append(path)
        for child in path.rglob("*"):
            if child.is_file() and not child.is_symlink() and child.suffix in ARTIFACT_SUFFIXES:
                candidates[child] = "retired_generated_output; configurations and metrics retained"
    skipped = []
    for path in list(candidates):
        if str(path.resolve()) in protected or any(path.is_relative_to(p) for p in protected_directories):
            skipped.append(str(path.relative_to(ROOT)))
            del candidates[path]
    cache, rows = {}, []
    for index, (path, reason) in enumerate(sorted(candidates.items())):
        local_file(path)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in cache:
            cache[identity] = digest(path)
        rows.append(dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size,
                         allocated_bytes=stat.st_blocks * 512, sha256=cache[identity],
                         device=stat.st_dev, inode=stat.st_ino, links=stat.st_nlink,
                         mtime_ns=stat.st_mtime_ns, reason=reason))
        if index % 100 == 0:
            print(f"Inventoried {index + 1}/{len(candidates)} files", flush=True)
    counts = Counter((r["device"], r["inode"]) for r in rows)
    freed, seen = 0, set()
    for row in rows:
        key = (row["device"], row["inode"])
        if key not in seen and counts[key] == row["links"]:
            freed += row["allocated_bytes"]
        seen.add(key)
    result = dict(protocol="explicit-retirement-v1", status="planned",
                  created_at=datetime.now(timezone.utc).isoformat(),
                  policy=str(policy_path.resolve()), policy_sha256=digest(policy_path),
                  source_bundle=policy["source_bundle"], protected=protected,
                  protected_directories=[str(p) for p in protected_directories],
                  output_roots=[str(p) for p in output_roots], skipped_protected=skipped,
                  explicit_artifacts=[str(local_file(p).relative_to(ROOT)) for p in explicit_artifacts],
                  files=rows, file_count=len(rows), apparent_bytes=sum(r["bytes"] for r in rows),
                  estimated_released_bytes=freed, free_disk_before=shutil.disk_usage(ROOT).free)
    write_json(output, result)
    print(json.dumps({k: result[k] for k in ("file_count", "apparent_bytes", "estimated_released_bytes")}))


def apply(plan_path, report_path):
    plan_path = Path(plan_path)
    result = json.loads(plan_path.read_text())
    if result["protocol"] != "explicit-retirement-v1" or result["status"] != "planned":
        raise ValueError("Expected an unapplied, explicit cleanup plan")
    if digest(result["policy"]) != result["policy_sha256"]:
        raise ValueError("Cleanup policy changed")
    bundle = result["source_bundle"]
    if digest(ROOT / bundle["path"]) != bundle["sha256"]:
        raise ValueError("Source recovery bundle changed")
    cache = {}
    def checked_hash(path):
        s = path.stat()
        key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if key not in cache:
            cache[key] = digest(path)
        return cache[key]
    for value, sha in result["protected"].items():
        if checked_hash(Path(value)) != sha:
            raise ValueError(f"Protected dependency changed: {value}")
    # Validate the complete plan before performing its first mutation.
    for row in result["files"]:
        path = local_file(row["path"])
        if str(path) in result["protected"] or any(path.is_relative_to(Path(p)) for p in result["protected_directories"]):
            raise ValueError(f"Protected path in cleanup plan: {path}")
        permitted = (path.parent == BASE and path.suffix == ".py") or (
            path.suffix in ARTIFACT_SUFFIXES and (any(path.is_relative_to(Path(p)) for p in result["output_roots"]) or
                (row["path"] in result.get("explicit_artifacts", []) and path.is_relative_to(ROOT / "runs"))))
        if not permitted or checked_hash(path) != row["sha256"]:
            raise ValueError(f"Cleanup candidate changed or is outside scope: {path}")
    deleted = []
    try:
        for row in result["files"]:
            path = local_file(row["path"])
            stat = path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                row["device"], row["inode"], row["bytes"], row["mtime_ns"]):
                raise ValueError(f"Candidate changed during cleanup: {path}")
            path.unlink()
            deleted.append(row)
    finally:
        write_json(report_path, dict(protocol=result["protocol"], status="executed" if len(deleted) == len(result["files"]) else "partial",
            plan=str(plan_path.resolve()), plan_sha256=digest(plan_path),
            deleted=deleted, deleted_count=len(deleted), source_bundle=bundle,
            protected_files_verified=len(result["protected"]),
            free_disk_before=result["free_disk_before"], free_disk_after=shutil.disk_usage(ROOT).free,
            historical_evidence="Retired reports and audits describe pre-cleanup assets; use the relocation index and source bundle for recovery."))
    print(f"Removed {len(deleted)} explicitly retired files; protected dependencies verified.")


def verify(plan_path, report_path, output):
    """Recheck retained dependencies after cleanup without loading checkpoints."""
    plan_value = json.loads(Path(plan_path).read_text())
    report = json.loads(Path(report_path).read_text())
    if report['status'] != 'executed' or report['plan_sha256'] != digest(plan_path):
        raise ValueError('Execution record does not match this completed plan')
    for path, sha in plan_value['protected'].items():
        if digest(path) != sha:
            raise ValueError(f'Retained dependency changed: {path}')
    remaining = [r['path'] for r in plan_value['files'] if (ROOT / r['path']).exists()]
    if remaining:
        raise ValueError(f'Retired files remain: {remaining[:5]}')
    bundle = plan_value['source_bundle']
    if digest(ROOT / bundle['path']) != bundle['sha256']:
        raise ValueError('Source recovery bundle changed')
    summary = dict(status='verified', checked_at=datetime.now(timezone.utc).isoformat(),
                   removed_files=report['deleted_count'],
                   retired_source_files=len([r for r in plan_value['files'] if r['path'].endswith('.py')]),
                   estimated_released_bytes=plan_value['estimated_released_bytes'],
                   protected_files_verified=len(plan_value['protected']), source_bundle=bundle,
                   plan_sha256=digest(plan_path), execution_sha256=digest(report_path),
                   free_disk_after=shutil.disk_usage(ROOT).free)
    write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("apply")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p = commands.add_parser("verify")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.policy, args.output)
    elif args.command == "apply":
        apply(args.plan, args.report)
    else:
        verify(args.plan, args.report, args.output)


if __name__ == "__main__":
    main()
