"""One command-line entry point for experiment monitoring and review tools."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time


BASE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = BASE / "runs/native30_gait_v2_ego55k_oct15"
COMMANDS = {
    "rich-test": ("rich_test", "准备独立 RICH TEST、冻结模型评估和 CPU 复算"),
    "adapt": ("adapt", "从选定EGO扩散权重进行有预算的EXO适配并保留EGO replay"),
    "view": ("motion_viewer", "查看已有预测、GT 和场景，不运行模型推理"),
    "best": ("checkpoint_guardian", "按固定验证指标保留一个最佳模型副本"),
    "scene-check": ("scene_causal_check", "运行已登记的冻结权重场景/物理修正配对检查"),
    "motion-check": ("motion_ablation", "运行冻结权重的头部锚定、重力方向和支撑配对实验"),
    "support-check": ("contact_projection", "仅用 CPU 检查已有输出的可见支撑关节投影"),
    "motion-trace": ("motion_diagnostics", "记录冻结权重采样中各阶段的脚部误差"),
    "foot-trial": ("paired_foot_trial", "从相同模型、Adam和随机状态配对短训，验证独立物理修正的影响"),
    "reference-trial": ("reference_trial", "复用匹配对照，短训检验头部水平朝向参考"),
}


def read_status(run: Path) -> dict:
    result = {"run": str(run.resolve())}
    for name, relative in (
        ("training", "continuous_training_oct16/STATUS.json"),
        ("best", "training/BEST_GUARDIAN_STATUS.json"),
    ):
        path = run / relative
        result[name] = json.loads(path.read_text()) if path.exists() else None
    current = run / 'training/STATUS.json'
    if result['training'] is None and current.exists():
        result['training'] = json.loads(current.read_text())
    from scenemi.monitoring import progress_snapshot, public_progress
    progress = progress_snapshot(run)
    if progress is not None:
        result['observed_progress'] = public_progress(progress)
        result['observed_progress']['note'] = (
            'Logged updates may exceed the last checkpoint; log activity alone '
            'does not prove process liveness. Use --sample-seconds for measured throughput.')
    pipeline = run / 'STATUS.json'
    if pipeline.exists():
        result['pipeline'] = json.loads(pipeline.read_text())
    prepared = run / 'PREPARED_CONTRACT.json'
    if prepared.exists():
        config = json.loads(prepared.read_text()).get('config', {})
        variants = config.get('variants', [])
        if variants:
            result['execution'] = 'sequential: train and evaluate each variant before the next'
            result['variant_order'] = variants
            result['variants'] = {}
            for variant in variants:
                path = run / variant / 'STATUS.json'
                result['variants'][variant] = (json.loads(path.read_text()) if path.exists() else
                    dict(state='not_started', completed_updates=0,
                         planned_updates=config['training']['planned_updates']))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="读取训练和 best 状态")
    status.add_argument("--run", type=Path, default=DEFAULT_RUN)
    status.add_argument('--sample-seconds', type=float, default=0,
                        help='Measure new log updates over 0–60 seconds using a monotonic clock')
    for name, (_, help_text) in COMMANDS.items():
        # The existing tool owns its options, including --help. Importing it
        # only after dispatch keeps status/help free of torch and CUDA imports.
        commands.add_parser(name, add_help=False, help=help_text)
    args, remaining = parser.parse_known_args(argv)
    if args.command == "status":
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        if not 0 <= args.sample_seconds <= 60:
            parser.error('--sample-seconds must be between 0 and 60')
        measurement = None
        if args.sample_seconds:
            from scenemi.monitoring import progress_snapshot, measure_progress
            before = progress_snapshot(args.run)
            time.sleep(args.sample_seconds)
            after = progress_snapshot(args.run)
            measurement = (before, after)
        result = read_status(args.run)
        if measurement is not None:
            planned = (result.get('training') or {}).get('planned_updates')
            result['measured_progress'] = measure_progress(*measurement, planned_updates=planned)
        print(json.dumps(result, ensure_ascii=False, indent=2))
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
