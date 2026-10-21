"""Admit and bound only our jobs on a machine shared with other WSL users.

This is a launch-time check, not a reservation of the shared GPU or a kernel
crash fix. It never stops processes, changes WSL settings, or starts CUDA.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid


GIB = 2 ** 30
MODULE = "experiments.offline_camera_retrain_v1.ops.resources"


def inspect_gpu(query=subprocess.check_output):
    common = ["nvidia-smi", "--id=0", "--format=csv,noheader,nounits"]
    raw = query(common + ["--query-gpu=memory.used,utilization.gpu"],
                text=True, timeout=10)
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("Expected one device-zero GPU row")
    used, utilization = [int(value.strip()) for value in rows[0].split(",")]
    if used < 0 or not 0 <= utilization <= 100:
        raise ValueError("Invalid global GPU measurement")
    raw = query(common + ["--query-compute-apps=pid"], text=True, timeout=10)
    pids = [int(line.strip()) for line in raw.splitlines() if line.strip()]
    if any(pid <= 0 for pid in pids):
        raise ValueError("Invalid global GPU process measurement")
    return dict(memory_used_mib=used, utilization_percent=utilization,
                compute_pids=pids,
                admitted=not pids and used <= 2048 and utilization <= 10,
                policy="No compute PID; at most 2048MiB and 10% global usage")


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable is unavailable")


def admission(device, memory_gib):
    free = available_memory()
    report = dict(device=device, available_memory_bytes=free,
                  memory_max_bytes=int(memory_gib * GIB),
                  shared_memory_reserve_bytes=4 * GIB,
                  reservation=False, other_processes_modified=False)
    report["memory_admitted"] = free >= report["memory_max_bytes"] + 4 * GIB
    if device == "cuda":
        report["gpu"] = inspect_gpu()
    report["admitted"] = report["memory_admitted"] and (
        device == "cpu" or report["gpu"]["admitted"])
    return report


def scope_command(command, memory_gib, cpu_count):
    maximum = int(memory_gib * GIB)
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect",
            "--unit=scenemi-" + uuid.uuid4().hex[:12],
            "--property=MemoryHigh=" + str(int(maximum * .8)),
            "--property=MemoryMax=" + str(maximum),
            "--property=MemorySwapMax=" + str(GIB),
            "--property=TasksMax=256",
            "--", sys.executable, "-m", MODULE, "_exec",
            "--memory-gib", str(memory_gib), "--cpu-count", str(cpu_count),
            "--", *command]


def bounded_exec(command, memory_gib, cpu_count):
    """Verify effective limits before starting the actual heavy program."""
    relative = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limits = {name: (group / name).read_text().strip()
              for name in ("memory.max", "memory.high", "memory.swap.max", "pids.max")}
    maximum = int(memory_gib * GIB)
    for name, expected in (("memory.max", maximum),
                           ("memory.high", int(maximum * .8)),
                           ("memory.swap.max", GIB), ("pids.max", 256)):
        if limits[name] == "max" or int(limits[name]) > expected:
            raise RuntimeError("Required cgroup limit not effective: " + name)
    # WSL user units here lack the CPU controller. Affinity bounds this job
    # without enabling controllers or editing any parent/shared cgroup.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[-cpu_count:])
    limits["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if len(limits["cpu_affinity"]) > cpu_count:
        raise RuntimeError("Required CPU affinity limit not effective")
    print(json.dumps(dict(effective_job_limits=limits), ensure_ascii=False), flush=True)
    os.execvpe(command[0], command, os.environ)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run", "_exec"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--memory-gib", type=float, default=12)
    parser.add_argument("--cpu-count", type=int, default=4)
    # Explicit '--' separates the bounded program from guard options.
    raw = list(sys.argv[1:] if argv is None else argv)
    separator = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:separator])
    command = raw[separator + 1:]
    if not all(math.isfinite(value) and value > 0
               for value in (args.memory_gib, args.cpu_count)):
        parser.error("Memory and CPU budgets must be positive")
    if (args.action != "check") != bool(command):
        parser.error("run requires a program after '--'; check takes no program")
    if args.action == "_exec":
        bounded_exec(command, args.memory_gib, args.cpu_count)
        return
    try:
        report = admission(args.device, args.memory_gib)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps(dict(admitted=False, error=str(exc)), ensure_ascii=False),
              flush=True)
        raise SystemExit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["admitted"]:
        print("Resource admission refused; no program was started.", file=sys.stderr)
        raise SystemExit(2)
    if args.action == "run":
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = str(max(1, int(args.cpu_count)))
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        # A failed systemd limit setup must never fall back to an unbounded job.
        result = subprocess.call(scope_command(command, args.memory_gib,
                                               args.cpu_count), env=env)
        raise SystemExit(result)


if __name__ == "__main__":
    main()
