"""One command-line entry point for experiment monitoring and review tools."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = BASE / "runs/native30_gait_v2_ego55k_oct15"
COMMANDS = {
    "view": ("motion_viewer", "查看已有预测、GT 和场景，不运行模型推理"),
    "best": ("checkpoint_guardian", "按固定验证指标保留一个最佳模型副本"),
    "scene-check": ("scene_causal_check", "运行已登记的冻结权重场景/物理修正配对检查"),
    "motion-check": ("motion_ablation", "运行冻结权重的头部锚定、重力方向和支撑配对实验"),
    "support-check": ("contact_projection", "仅用 CPU 检查已有输出的可见支撑关节投影"),
    "motion-trace": ("motion_diagnostics", "记录冻结权重采样中各阶段的脚部误差"),
    "foot-trial": ("paired_foot_trial", "从相同模型、Adam和随机状态配对短训，验证独立物理修正的影响"),
}


def read_status(run: Path) -> dict:
    result = {"run": str(run.resolve())}
    for name, relative in (
        ("training", "continuous_training_oct16/STATUS.json"),
        ("best", "training/BEST_GUARDIAN_STATUS.json"),
    ):
        path = run / relative
        result[name] = json.loads(path.read_text()) if path.exists() else None
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="读取训练和 best 状态")
    status.add_argument("--run", type=Path, default=DEFAULT_RUN)
    for name, (_, help_text) in COMMANDS.items():
        # The existing tool owns its options, including --help. Importing it
        # only after dispatch keeps status/help free of torch and CUDA imports.
        commands.add_parser(name, add_help=False, help=help_text)
    args, remaining = parser.parse_known_args(argv)
    if args.command == "status":
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        print(json.dumps(read_status(args.run), ensure_ascii=False, indent=2))
        return
    module_name = COMMANDS[args.command][0]
    module = importlib.import_module(f"{__package__}.{module_name}")
    previous = sys.argv
    sys.argv = [f"{parser.prog} {args.command}", *remaining]
    try:
        module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
