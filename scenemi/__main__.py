"""Stable SceneMI command line. Heavy jobs use the existing resource wrapper."""
import argparse
import json
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG, load_config, load_paths


def resource_main(argv):
    from .execution import resource_main as run
    return run(argv)




def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == 'run':
        from .launch import main as launch_main
        return launch_main(raw[1:])
    if raw and raw[0] == 'resources':
        return resource_main(raw[1:])
    if raw and raw[0] == 'egogvae':
        if raw[1:2] == ['run']:
            from .egogvae_execution import main as baseline_main
            return baseline_main(raw[2:])
        from .egogvae_adaptation import main as baseline_main
        return baseline_main(raw[1:])
    if raw and raw[0] in ('train', 'preflight'):
        # Stable public entry point for the four-corpus trainer.  The legacy
        # module still owns the implementation and checkpoint format.
        from .train import main as training_main
        return training_main(raw)
    if raw and raw[0] == 'evaluate':
        if '--config' in raw:
            import yaml
            file = Path(raw[raw.index('--config')+1])
            if yaml.safe_load(file.read_text()).get('version') == 'ego-scene-refinement-v1':
                from .refinement.training import main as refinement_main
                return refinement_main(raw)
        from .evaluate import main as evaluate_main
        return evaluate_main(raw[1:])
    if raw and raw[0] == 'exo-camera':
        from .data.exo_export import main as exo_main
        return exo_main(raw[1:])
    # Delegate existing operational tools without copying their implementations.
    if raw and raw[0] == 'ops':
        from experiments.offline_camera_retrain_v1.ops.__main__ import main as ops_main
        return ops_main(raw[1:])
    if raw and raw[0] in ('view', 'visualize', 'viser'):
        from .visualization import main as view_main
        return view_main(raw[1:])
    if raw and raw[0] == 'grail':
        from .data.grail_quality import main as grail_main
        return grail_main(raw[1:])
    if raw and raw[0] == 'gimo':
        if raw[1:2] == ['poses']:
            from .data.gimo_pose import main as gimo_pose_main
            return gimo_pose_main(raw[2:])
        from .data.gimo import main as gimo_main
        return gimo_main(raw[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('run', help='Run training from an explicit YAML configuration')
    commands.add_parser('train', help='Train EGO-only from a YAML configuration (manual launch)')
    commands.add_parser('preflight', help='Check a real EGO training update under resource limits')
    commands.add_parser('evaluate', help='Evaluate a saved EGO checkpoint on fixed validation windows')
    commands.add_parser('view', aliases=['viser', 'visualize'], help='Shared Viser playback and source previews')
    commands.add_parser('ops', help='Existing monitoring, review and resource tools')
    commands.add_parser('resources', help='Check resources or run a command under RAM/CPU limits')
    commands.add_parser('grail', help='GRAIL data quality tools')
    commands.add_parser('gimo', help='GIMO source audit, EGO preprocessing and Viser review')
    commands.add_parser('exo-camera', help='EXO camera preprocessing tools')
    commands.add_parser('egogvae', help='Prepare, evaluate or manually fine-tune the public EgoGVAE')
    config = commands.add_parser('config', help='Validate the 128-frame AMASS stage-A configuration')
    config.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    amass = commands.add_parser('amass', help='Audit raw AMASS and the pinned EgoAllo protocol')
    actions = amass.add_subparsers(dest='action', required=True)
    inventory = actions.add_parser('inventory')
    inventory.add_argument('--paths', type=Path, required=True)
    inventory.add_argument('--output', type=Path, required=True)
    plan = actions.add_parser('plan')
    plan.add_argument('--paths', type=Path, required=True)
    plan.add_argument('--python', type=Path, required=True, help='Python in an isolated EgoAllo environment')
    prepare = actions.add_parser('prepare', help='Run and audit upstream 0a on CPU; resumable by content hashes')
    prepare.add_argument('--paths', type=Path, required=True)
    prepare.add_argument('--inventory', type=Path, required=True)
    prepare.add_argument('--pilot', action='store_true', help='First obtain one eligible motion per corpus')
    prepare.add_argument('--bounded-height-clustering', action='store_true',
                         help='Use the verified linear-memory height-clustering execution adapter')
    verify_heights = actions.add_parser('verify-height-clustering',
                                       help='Compare bounded height clustering with original real-data outputs')
    verify_heights.add_argument('--paths', type=Path, required=True)
    index = actions.add_parser('index', help='Verify and freeze the prepared data index')
    index.add_argument('--paths', type=Path, required=True)
    index.add_argument('--diagnostic', action='store_true', help='Pilot only; forbidden for formal training')
    for name in ('preflight', 'train', 'ready'):
        entry = actions.add_parser(name)
        entry.add_argument('--paths', type=Path, required=True)
        entry.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        if name != 'ready':
            entry.add_argument('--device', choices=('cpu', 'cuda'), default='cpu' if name=='preflight' else 'cuda')
            entry.add_argument('--index', type=Path)
        if name == 'preflight':
            entry.add_argument('--diagnostic', action='store_true')
        if name == 'train':
            entry.add_argument('--resume', action='store_true')
        if name == 'ready':
            entry.add_argument('--wait-for-data', action='store_true')
    args = parser.parse_args(raw)
    if args.command == 'config':
        config = load_config(args.config)
        result = dict(valid=True, config=str(args.config.resolve()), frames=config['data']['frames'],
                      fps=config['data']['fps'], scene_enabled=config['scene']['enabled'],
                      initialization=config['model']['initialization'], modes=config['observation']['mode_probabilities'],
                      implementation=config['training']['implementation'],
                      readiness='Run amass ready; configuration validity is separate from runtime readiness')
    else:
        from .data.amass import inventory, preprocessing_plan
        paths = load_paths(args.paths)
        if args.action == 'prepare':
            if args.bounded_height_clustering:
                if args.pilot:
                    parser.error('The bounded execution adapter resumes the audited full dataset')
                from .data.amass_prepare_bounded import prepare
                result = prepare(paths, args.inventory)
            else:
                from .data.amass_prepare import prepare
                result = prepare(paths, args.inventory, pilot=args.pilot)
        elif args.action == 'verify-height-clustering':
            from .data.amass_prepare_bounded import verify
            result = verify(paths)
        elif args.action == 'index':
            from .data.amass_dataset import build_index
            result = dict(index=str(build_index(paths, diagnostic=args.diagnostic)))
        elif args.action in ('preflight', 'train', 'ready'):
            from .training import preflight, train, readiness
            config = load_config(args.config)
            if args.action == 'ready':
                result = readiness(paths, config, wait=args.wait_for_data)
            elif args.action == 'preflight':
                checked = preflight(paths, config, index=args.index, device=args.device, diagnostic=args.diagnostic)
                result = {key: checked[key] for key in ('state', 'device', 'diagnostic_only',
                          'batch_size', 'updates_executed', 'resume_bitwise_exact',
                          'isolation', 'elapsed_seconds', 'peak_cuda_bytes')}
                result['full_record'] = str(paths['output']/'preflight'/
                    (('diagnostic_' if args.diagnostic else 'complete_')+args.device)/'RESULT.json')
            else:
                result = train(paths, config, index=args.index, device=args.device, resume=args.resume)
        else:
            result = inventory(paths, args.output) if args.action == 'inventory' else preprocessing_plan(paths, args.python)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
