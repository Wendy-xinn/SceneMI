"""Train the adapted SceneMI AdaGN U-Net from random initialization."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import GROUPS, LENGTHS, RICH_SOURCES, SOURCE, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.control import MODE_PROBABILITIES, fixed_control_mask, sample_control_masks
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.typed_head_condition import TypedSceneMI as OfflineSceneMI, observation_loss
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation as prepare_observation
from experiments.offline_camera_retrain_v1.staged_head_optimization import set_training_phase, optimizer_groups, update_learning_rates
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.gallery import write_run_index
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

HERE = Path(__file__).resolve().parent


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audited_data_fingerprint(rich_source='legacy5interp'):
    ready = json.loads((SOURCE / 'READY.json').read_text())
    if ready.get('status') != 'ready':
        raise ValueError('Original sequence archive is not audited')
    source_paths = [SOURCE / 'sequences.jsonl', SOURCE / 'intervals.jsonl']
    source_paths += [SOURCE / 'trumans_sequences' / f'{s}.jsonl' for s in ('train', 'validation')]
    source_paths += [SOURCE / 'egobody_sequences' / f'{s}.jsonl' for s in ('train', 'validation')]
    if fingerprint(source_paths) != ready['manifest_sha256']:
        raise ValueError('Original sequence manifests changed after audit')
    fk = json.loads((HERE / 'data/poses/fk_audit.json').read_text())
    if any(item['p95_m'] > .08 for split in fk.values() for item in split.values()):
        raise ValueError('FK alignment failed')
    pose_paths = [HERE / 'data/poses' / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    rich_paths = []
    rich_names = (('rich_train_smpl20', 'rich_val_smpl20')
                  if rich_source == 'legacy5interp' else
                  ('rich_train_smpl_native20_faceout_oct07',
                   'rich_val_smpl_native20_faceout_oct07'))
    for rich_name in rich_names:
        root = HERE / 'data/scene_visibility_v2_oct05' / rich_name
        rich_paths.extend(sorted(root.glob('*/metadata.json')))
    if len(rich_paths) != 90:
        raise ValueError('Canonical RICH train/validation bundles are incomplete')
    if rich_source != 'legacy5interp':
        for path in rich_paths:
            meta = json.loads(path.read_text())
            if (meta.get('source_stride_for_motion') != 1
                    or meta.get('source_fps') != 30.0
                    or meta.get('output_fps') != 20.0
                    or 'identity face mount' not in meta.get('camera_protocol', '')
                    or meta.get('fk_p95_m', 1.) > .005
                    or meta.get('approved_camera_angle_max_deg', 180.) > .1
                    or meta.get('face_normal_dot', -1.) < .5
                    or meta.get('full_wearer_first_hit_fraction', 1.) >= .5):
                raise ValueError(f'{path}: RICH native-motion/camera audit failed')
    visible_paths = []
    for dataset, visible_name, source_name in (
            ('TRUMANS', 'visible_trumans_5hz', 'trumans_sequences'),
            ('EgoBody', 'visible_egobody_5hz', 'egobody_sequences')):
        for split in ('train', 'validation', 'test'):
            path = HERE / 'data' / visible_name / f'{split}.jsonl'
            visible_paths.append(path)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            expected = len((SOURCE / source_name / f'{split}.jsonl').read_text().splitlines())
            metadata_valid = all(
                row.get('sample_stride') == 4
                and row.get('candidate_count') == 65536
                and row.get('query_radius_m') == 6.5
                and 'project' in row.get('protocol', '').lower()
                for row in rows)
            if (len(rows) != expected or not metadata_valid
                    or not all(Path(row['path']).is_file() for row in rows)):
                raise ValueError(f'{dataset}/{split}: incomplete or invalid 5 Hz ego-visible scene map')
    return fingerprint(source_paths + pose_paths + visible_paths + rich_paths + [HERE / 'data/poses/fk_audit.json'])


def sample_batch(dataset, length, group, size,return_identities=False):
    pairs=[dataset.sample(length,group) for _ in range(size)]
    batch=collate([sample for sample,identity in pairs])
    return (batch,[identity for sample,identity in pairs]) if return_identities else batch


def balanced_schedule(steps, seed):
    """Exact three-dataset balance and enough RICH 64-frame draws for coverage."""
    counts = {
        'trumans': (12, 18, 10),
        'camera_wearer': (6, 9, 5),
        'interactee': (6, 9, 5),
        'rich': (12, 18, 10),
    }
    # 120 steps: TRUMANS 40, EgoBody roles 20 each, RICH 40.
    # Lengths within each domain are 30%/45%/25%.
    block = [(group, length) for group, per_length in counts.items()
             for length, n in zip(LENGTHS, per_length) for _ in range(n)]
    rng = np.random.default_rng(seed + 7001)
    schedule = []
    while len(schedule) < steps:
        indices = rng.permutation(len(block))
        schedule.extend(block[i] for i in indices)
    return schedule[:steps]


def to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'runs/mixed_offline_scenemi_v1')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--validation-samples', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--scene-dropout', type=float, default=.1)
    parser.add_argument('--disable-scene', action='store_true')
    parser.add_argument('--ddim-every', type=int, default=0,
                        help='Run sampling and write a gallery at this checkpoint interval (0 disables)')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--source-recovery-audit',type=Path,help='Explicit audited metadata-recovery transfer for warm start only; resume still requires exact current source hash')
    parser.add_argument('--init-from', type=Path,
                        help='Warm-start weights only; reset optimizer, schedule and RNG for a paired trial')
    parser.add_argument('--loss-profile', choices=('baseline', 'gait_v1', 'gait_v2', 'coordination_v1', 'orientation_v1', 'orientation_v2', 'orientation_v3', 'orientation_v4', 'orientation_v5'), default='baseline')
    parser.add_argument('--hard-head-rotation', action='store_true', help='Experimental calibrated synthetic-camera rotation projection in training; real PV excluded; inference must use the same projection')
    parser.add_argument('--ddim-head-only', action='store_true')
    parser.add_argument('--allow-legacy-scene-for-diagnostics',action='store_true',help='Explicitly permit old scene inputs for diagnostic trials only')
    parser.add_argument('--temporal-scene-manifest',type=Path,help='Complete exact20 RICH/EgoBody scene manifest; requires native_v1')
    parser.add_argument('--rich-causal-scene',action='store_true',help='Timestamp-gated 5Hz RICH observations with recording history')
    parser.add_argument('--rich-contact-root',type=Path,help='Native20 packed contact labels; requires native_v1')
    parser.add_argument('--contact-loss-weight',type=float,default=.1)
    parser.add_argument('--body-protocol',choices=('legacy','native_v1'),default='legacy',help='native_v1: native SMPL/SMPL-X 22-body supervision and shape/model conditioning')
    parser.add_argument('--skeleton-profile', choices=('archived', 'trumans_male_v2', 'canonical_smpl'), default='archived')
    parser.add_argument('--domain-duration-alpha', type=float, help='Use unique eligible duration^alpha weighting, alpha in [0,1]; omitted retains legacy equal schedule')
    parser.add_argument('--trumans-window-protocol',choices=('legacy_stable_v1','temporal_valid_v1'),help='Versioned eligibility: temporal_valid_v1 includes moving objects, retaining source validity checks')
    parser.add_argument('--trumans-scene-manifest', type=Path, help='Full-recording time-resolved v2 visibility bundles; missing coverage is fatal')
    parser.add_argument('--rich-source', choices=RICH_SOURCES, default='legacy5interp')
    parser.add_argument('--precision', choices=('bf16', 'float32'), default='bf16')
    parser.add_argument('--low-noise-fraction', type=float, default=0.,
                        help='Fraction of additional low-noise t=0..99 denoising examples')
    parser.add_argument('--keep-inference-every', type=int, default=5000,
                        help='Keep model-only weights at this sampling interval, plus the final step')
    parser.add_argument('--observation-protocol', choices=('joint','rotation_only','mild'), required=True)
    parser.add_argument('--encoder-only-steps',type=int,default=200)
    parser.add_argument('--warmup-steps',type=int,default=100)
    args = parser.parse_args()
    if not 0<=args.encoder_only_steps<args.steps or args.warmup_steps<1:
        parser.error('Invalid encoder phase/warmup budget')
    observation_rng = torch.Generator(device='cpu').manual_seed(args.seed + 19731)
    if args.hard_head_rotation or args.ddim_every:
        parser.error('Typed trials use soft conditions and their versioned evaluator')
    if args.domain_duration_alpha is not None and not 0 <= args.domain_duration_alpha <= 1:
        parser.error('domain-duration-alpha must be in [0,1]')
    if args.hard_head_rotation and args.body_protocol != 'native_v1':
        parser.error('hard-head-rotation requires native_v1 known virtual-camera mounts')
    if args.resume and args.init_from:
        parser.error('resume and init-from are mutually exclusive')
    if not 0 <= args.low_noise_fraction <= 1:
        parser.error('low-noise-fraction must be in [0, 1]')
    if args.keep_inference_every < 1:
        parser.error('keep-inference-every must be positive')
    if (args.steps < 1 or args.batch_size < 1 or args.save_every < 1
            or args.validation_samples < 1):
        parser.error('steps, batch-size, save-every and validation-samples must be positive')
    if not 0 <= args.scene_dropout <= 1:
        parser.error('scene-dropout must be in [0, 1]')
    if args.ddim_every < 0 or (args.ddim_every and args.ddim_every % args.save_every):
        parser.error('ddim-every must be zero or a positive multiple of save-every')
    if not torch.cuda.is_available():
        raise RuntimeError('SceneMI training requires a CUDA GPU')
    if any((args.output / name).exists() for name in ('last.pt', 'best.pt', 'training_log.jsonl')) and not args.resume:
        raise FileExistsError('Checkpoint exists; use --resume or another output directory')
    if args.body_protocol=='native_v1' and not args.allow_legacy_scene_for_diagnostics and (not args.temporal_scene_manifest or not args.trumans_scene_manifest):parser.error('native_v1 training requires complete exact20 scenes for all three datasets; legacy inputs require explicit diagnostic flag')
    if args.temporal_scene_manifest and args.body_protocol!='native_v1':parser.error('Temporal native scenes require native_v1')
    if args.rich_causal_scene and args.body_protocol!='native_v1':parser.error('Causal RICH requires native_v1')
    if args.rich_contact_root and args.body_protocol!='native_v1':parser.error('Contact supervision requires native_v1')
    if args.contact_loss_weight<0:parser.error('Contact weight must be nonnegative')
    if args.body_protocol == 'native_v1':
        args.rich_source = 'native20_faceout_oct07'
        args.skeleton_profile = 'trumans_male_v2'
    if args.trumans_window_protocol is None:args.trumans_window_protocol='temporal_valid_v1' if args.body_protocol=='native_v1' and args.trumans_scene_manifest and args.temporal_scene_manifest else 'legacy_stable_v1'
    data_hash = audited_data_fingerprint(args.rich_source)
    if args.trumans_window_protocol=='temporal_valid_v1':data_hash=hashlib.sha256((data_hash+fingerprint([HERE/'temporal_windows.py',HERE.parents[2]/'TRUMANS/bad_frames.npy'])).encode()).hexdigest()
    if args.body_protocol == 'native_v1':
        native_paths=[HERE/'native_body_data.py',HERE/'scene_model.py']
        for split in ('train','validation'):
            for row in map(json.loads,(HERE/'data/poses'/f'{split}.jsonl').read_text().splitlines()):
                folder=Path(row['folder']);native_paths.extend(folder/name for name in ['pose_axis_angle_world.npy','translation_world.npy'])
                if row['dataset']=='trumans':native_paths.append(Path(row['prepared_folder'])/'joints_world.npy')
        for gender in ('MALE','FEMALE'):
            native_paths.extend([HERE.parents[2]/'SceneMI/body_models/smplx'/f'SMPLX_{gender}.npz',HERE.parents[2]/'ProtoMotions/data/smpl'/f'SMPL_{gender}.pkl'])
        for split in ('train','val'):
            native_paths.extend(sorted((HERE.parents[2]/'RICH/extracted'/f'{split}_body').glob('*/*/*.pkl')))
        data_hash=hashlib.sha256((data_hash+fingerprint(native_paths)).encode()).hexdigest()
    if args.trumans_scene_manifest:
        manifest_rows = [json.loads(line) for line in args.trumans_scene_manifest.read_text().splitlines() if line]
        scene_paths = [args.trumans_scene_manifest,HERE.parents[2]/'TRUMANS/processed/scene_expert_v1/clips.jsonl',HERE/'scene_visibility_v2.py',HERE/'causal_scene.py']
        for row in manifest_rows:
            if row['split'] not in ('train','validation'): continue
            if row.get('status')=='excluded':continue
            folder = Path(row['path'])
            if not folder.is_absolute(): folder = args.trumans_scene_manifest.resolve().parent / folder
            for f in ['causal_bps_20.npy','causal_bps_valid_20.npy']:scene_paths.append(folder/f)
            mm=Path(json.loads((folder/'metadata.json').read_text())['memory_bundle'])
            scene_paths.extend(mm/f for f in ['metadata.json','static_points.npy','first_observed_source_frames.npy'])
            if (folder/'observation_valid.npy').exists():scene_paths.append(folder/'observation_valid.npy')
            scene_paths.extend(folder/name for name in ['metadata.json','source_frame_ids.npy','visible_frame_points_scenemi_yup.npy','visible_frame_owner.npy','camera_position_scenemi_yup.npy','camera_rotation_scenemi_yup.npy'])
        data_hash = hashlib.sha256((data_hash + fingerprint(scene_paths)).encode()).hexdigest()
    if args.rich_contact_root:
        contact_paths=sorted(args.rich_contact_root.glob('*/*/contacts.npz'))
        if len(contact_paths)!=90:raise ValueError('Require all 90 native20 contact bundles')
        data_hash=hashlib.sha256((data_hash+fingerprint(contact_paths+[HERE/'contact_supervision.py'])).encode()).hexdigest()
    if args.temporal_scene_manifest:
        scene_paths=[args.temporal_scene_manifest,HERE/'causal_scene.py']
        for row in map(json.loads,args.temporal_scene_manifest.read_text().splitlines()):
            folder=Path(row['path'])
            if not folder.is_absolute():folder=args.temporal_scene_manifest.resolve().parent/folder
            mm=Path(json.loads((folder/'metadata.json').read_text())['memory_bundle'])
            scene_paths.extend(mm/f for f in ['metadata.json','static_points.npy','first_observed_source_frames.npy'])
            scene_paths.extend(folder/f for f in ['causal_bps_20.npy','causal_bps_valid_20.npy'])
            scene_paths.append(folder/'camera_intrinsics_128x96.npy')
            scene_paths.extend(folder/f for f in ['metadata.json','source_frame_ids.npy','visible_frame_points_scenemi_yup.npy','visible_frame_owner.npy','camera_position_scenemi_yup.npy','camera_rotation_scenemi_yup.npy'])
        data_hash=hashlib.sha256((data_hash+fingerprint(scene_paths)).encode()).hexdigest()
    if args.rich_causal_scene and not args.temporal_scene_manifest:
        scene_paths=[HERE/'causal_scene.py']
        for split in ('train','val'):
            for meta_path in sorted((HERE/'data/scene_visibility_v2_oct05'/f'rich_{split}_smpl_native20_faceout_oct07').glob('*/metadata.json')):
                folder=Path(json.loads(meta_path.read_text())['scene_bundle'])
                scene_paths.extend(folder/f for f in ('source_frame_ids.npy','visible_frame_points_scenemi_yup.npy','visible_frame_mask.npy'))
        data_hash=hashlib.sha256((data_hash+fingerprint(scene_paths)).encode()).hexdigest()
    if args.body_protocol=='native_v1' and not args.allow_legacy_scene_for_diagnostics:
        subprocess.run([sys.executable,str(HERE/'verify_scene20_dataset.py'),'--require-complete'],check=True,stdout=subprocess.DEVNULL)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    data_class=OfflineSceneMIData
    if args.body_protocol=='native_v1':
        from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
        data_class=NativeBodyData
    contact_kwargs={'contact_root':args.rich_contact_root,'rich_causal_scene':args.rich_causal_scene,'temporal_scene_manifest':args.temporal_scene_manifest} if args.body_protocol=='native_v1' else {}
    train = data_class('train', seed=args.seed,
                               skeleton_profile=args.skeleton_profile,
                               rich_source=args.rich_source, trumans_scene_manifest=args.trumans_scene_manifest, trumans_window_protocol=args.trumans_window_protocol, window_sampling=('duration' if args.domain_duration_alpha is not None else 'coverage'),**contact_kwargs)
    validation = data_class('validation', seed=777,
                                    skeleton_profile=args.skeleton_profile,
                                    rich_source=args.rich_source, trumans_scene_manifest=args.trumans_scene_manifest, trumans_window_protocol=args.trumans_window_protocol,**contact_kwargs)
    sampling_plan = None
    if args.domain_duration_alpha is None:
        schedule = balanced_schedule(args.steps, args.seed)
    else:
        from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_schedule
        schedule,sampling_plan = duration_schedule(train,args.steps,args.seed,args.domain_duration_alpha)
    args.output.mkdir(parents=True,exist_ok=True)
    for split,dataset in [('train',train),('validation',validation)]:
        (args.output/f'{split}_trumans_windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dataset.trumans_window_records))
    from collections import Counter
    draw_counts = Counter(schedule)
    uncovered = {f'{group}/{length}': (train.summary()[group][str(length)]['starts'],
                                     draw_counts[(group, length)] * args.batch_size)
                 for group in GROUPS for length in LENGTHS
                 if draw_counts[(group, length)] * args.batch_size
                 < train.summary()[group][str(length)]['starts']}
    if uncovered and args.domain_duration_alpha is None:
        raise ValueError(f'Training schedule cannot cover all windows: {uncovered}')
    condition_rng = np.random.default_rng(args.seed + 104729)
    for group in GROUPS:
        for length in LENGTHS:
            if not train.summary()[group][str(length)]['sequences']:
                raise ValueError(f'Missing train {group}/{length}')
            if not validation.summary()[group][str(length)]['sequences']:
                raise ValueError(f'Missing validation {group}/{length}')
    if args.trumans_scene_manifest:
        config_scene_protocol = 'TRUMANS per-frame first-hit dynamic; recording causal static memory; target-start global occupancy'
    else:
        config_scene_protocol = 'legacy frozen TRUMANS objects and offline sequence union'
    train_scenes = {r['scene_family'] for members in train.base.groups.values() for r, _ in members}
    val_scenes = {r['scene_family'] for members in validation.base.groups.values() for r, _ in members}
    # Preserve RICH's released train/val sequence split. It shares both scan
    # families and subjects; unseen-scene/person evaluation is a separate split.
    # This scene-family gate applies to EgoBody/TRUMANS source manifests only.
    if train_scenes & val_scenes:
        raise ValueError('Scene-family leakage')
    model = OfflineSceneMI(args.latent_dim, tuple(args.dim_mults),body_conditioning=args.body_protocol=='native_v1',contact_prediction=bool(args.rich_contact_root)).cuda()
    optimizer = torch.optim.AdamW(optimizer_groups(model,args.lr), lr=args.lr, weight_decay=.01)
    alphas = cosine_alphas().cuda()
    config = dict(**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  body_conditioning=args.body_protocol=='native_v1',
                  contact_prediction=bool(args.rich_contact_root),
                  source_hash=data_hash, duration_sampling_plan=sampling_plan, expected_window_draw_deficits=uncovered, trumans_scene_protocol=config_scene_protocol, architecture='SceneMI original AdaGN TemporalUnet + ViT occupancy + BPS MLP',
                  motion_representation='SMPL(-X) root translation, 22 identity-centred residual 6D rotations, 22 global joints',
                  condition=('masked per-joint position/orientation trajectories (head-dominant mixture) '
                             '+ visible static map + camera-centered BPS'),
                  control_mode_probabilities=dict(MODE_PROBABILITIES),
                  scene_condition_enabled=not args.disable_scene,
                  scene_dropout_probability=args.scene_dropout,
                  contact_protocol=('22 native body-region labels; masked BCE on RICH only; no GT contact input; inferred after generated motion' if args.rich_contact_root else 'disabled'),
                  rich_scene_protocol=('exact20 first-hit causal recording prefix' if args.temporal_scene_manifest else ('causal 5Hz static observations; recording prefix and start-fixed occupancy' if args.rich_causal_scene else 'offline union')),
                  losses=('x0 + rotation matrix + FK joint/velocity/acceleration + emphasized head path '
                          '+ stance-foot stillness + direct/FK consistency'),
                  schedule_bucket_steps={f'{g}/{l}': n for (g,l),n in draw_counts.items()},
                  train_summary=train.summary(), validation_summary=validation.summary())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.skeleton_profile == 'trumans_male_v2':
        config['skeleton_template_sha256'] = fingerprint([HERE / 'data/body_templates/trumans_male_rest.npy'])
    objective_paths=[HERE/'supervision.py', HERE/'orientation_supervision.py', HERE/'typed_head_condition.py', HERE/'bounded_head_observation.py', HERE/'staged_head_optimization.py', HERE/'train_bounded_head.py', HERE/'train_typed_head.py']
    config['observation_schema']='typed14-v1: pos3/rot6/pos_mask/rot_mask/pos_conf/rot_conf/type'
    config['observation_simulator']='GT anatomical head simulator; training confidence stays one; exclusive 20pct quota per five steps; mild position norm <=1cm and yaw <=3deg; evaluation synthetic confidence not estimator-calibrated'
    if args.hard_head_rotation:
        objective_paths.extend([HERE/'head_constraint.py',HERE/'train.py'])
        config['head_constraint_protocol']='known synthetic identity mount, observed joint15 rotation only; real PV soft; training and DDIM x0 projection; no root position correction'
    config['objective_sha256'] = fingerprint(objective_paths)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    first = 0
    best = float('inf')
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location='cpu', weights_only=False)
        if checkpoint['config']['source_hash'] != data_hash:
            if args.source_recovery_audit is None:
                raise ValueError('Warm-start source data mismatch')
            from experiments.offline_camera_retrain_v1.source_recovery_contract import validate_source_recovery
            validate_source_recovery(args.source_recovery_audit,checkpoint['config']['source_hash'],args)
            config['source_recovery_audit_sha256']=fingerprint([args.source_recovery_audit])
            config['initial_source_hash']=checkpoint['config']['source_hash']
            config['source_transfer_reason']='explicit native metadata recovery; frozen model/GT replay passed; current source hash retained'
        elif args.source_recovery_audit is not None:
            raise ValueError('Recovery audit is only for explicit source-hash transfer; omit it for unchanged data')
        model.load_state_dict(checkpoint['model'])
        config['initial_checkpoint_step'] = checkpoint['step']
        config['initial_total_steps'] = (checkpoint['config'].get('initial_total_steps',
                                         checkpoint['config'].get('initial_checkpoint_step', 0))
                                         + checkpoint['step'])
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        del checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key, default in (('steps',None),('lr',None),('batch_size',None),('seed',None),('encoder_only_steps',None),('warmup_steps',None),('observation_protocol',None),('trumans_window_protocol','legacy_stable_v1'),('loss_profile', 'baseline'), ('precision', 'bf16'), ('low_noise_fraction', 0.), ('domain_duration_alpha', None), ('body_protocol','legacy'),('contact_loss_weight',.1),('rich_causal_scene',False),('temporal_scene_manifest',None),('allow_legacy_scene_for_diagnostics',False)):
            if checkpoint['config'].get(key, default) != getattr(args, key):
                raise ValueError(f'Resume changed {key}; pass stored protocol flags, or use init-from for a new trial')
        if checkpoint['config'].get('objective_sha256') and checkpoint['config']['objective_sha256'] != config['objective_sha256']:
            raise ValueError('Resume changed supervision code; use init-from for an explicit new trial')
        if checkpoint['config'].get('skeleton_profile', 'archived') != args.skeleton_profile:
            raise ValueError('Resume cannot change skeleton profile; use init-from for an explicit new trial')
        if checkpoint['config'].get('skeleton_template_sha256') != config.get('skeleton_template_sha256'):
            raise ValueError('Skeleton template changed since checkpoint')
        if checkpoint['config']['source_hash'] != data_hash:
            raise ValueError('Source data changed since checkpoint')
        for recovery_key in ('source_recovery_audit_sha256','initial_source_hash','source_transfer_reason','initial_checkpoint_step','initial_total_steps'):
            if recovery_key in checkpoint['config']:config[recovery_key]=checkpoint['config'][recovery_key]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        first = checkpoint['step']
        best = checkpoint['best_validation']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        train.base.rng.bit_generator.state = checkpoint['data_rng_state']
        if 'sampler_state' in checkpoint:
            train.load_sampler_state(checkpoint['sampler_state'])
        observation_rng.set_state(checkpoint['observation_rng_state'])
        if 'condition_rng_state' in checkpoint:
            condition_rng.bit_generator.state = checkpoint['condition_rng_state']
    fixed_validation = {
        (group, length, sample_index): to_cuda(sample_batch(validation, length, group, 1))
        for group in GROUPS for length in (128, 192)
        for sample_index in range(args.validation_samples)}
    log = (args.output / 'training_log.jsonl').open('a', buffering=1)
    start_time = time.time()

    def apply_head_constraint(prediction,batch,mask,group):
        if not args.hard_head_rotation or group=='camera_wearer':return prediction
        from experiments.offline_camera_retrain_v1.head_constraint import project_head_orientation
        return project_head_orientation(prediction,batch['camera'],mask[...,15])

    def validation_metrics():
        model.eval()
        buckets = {}
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(777)
            for (group, length, _), batch in fixed_validation.items():
                batch = prepare_observation(batch,args.observation_protocol)
                clean = batch['motion']
                timestep = torch.full((len(clean),), 500, dtype=torch.long, device='cuda')
                alpha = alphas[timestep][:, None, None]
                noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
                head_mask = fixed_control_mask(len(clean), length, 'head', clean.device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    predicted = model(noisy, timestep, batch, control_mask=head_mask,
                                      use_scene=not args.disable_scene)
                predicted=apply_head_constraint(predicted.float(),batch,head_mask,group)
                losses = supervised_losses(predicted.float(), clean, batch)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    without_scene = model(noisy, timestep, batch, control_mask=head_mask,
                                          use_scene=False)
                    without_control = model(noisy, timestep, batch, control_mask=head_mask,
                                            use_scene=not args.disable_scene, use_control=False)
                without_scene=apply_head_constraint(without_scene.float(),batch,head_mask,group)
                values = {
                    'fk_mpjpe_cm': losses['fk_mpjpe_m'] * 100,
                    'head_error_cm': losses['head_error_m'] * 100,
                    'foot_sliding_cm_per_frame': losses['foot_sliding_m_per_frame'] * 100,
                    'x0': losses['x0'],
                    'fk_velocity_mse': losses['fk_velocity_mse'],
                    'fk_acceleration_mse': losses['fk_acceleration_mse'],
                    'scene_effect_l1': (predicted - without_scene).abs().mean(),
                    'control_effect_l1': (predicted - without_control).abs().mean(),
                }
                if args.rich_contact_root:
                    from experiments.offline_camera_retrain_v1.contact_supervision import contact_loss,contact_metrics
                    logits=model.contact_logits(predicted.float(),batch,use_scene=not args.disable_scene)
                    values.update(contact_bce=contact_loss(logits,batch),**contact_metrics(logits,batch))
                for metric, value in values.items():
                    buckets.setdefault((group, length, metric), []).append(float(value))
        model.train()
        return {f'{group}/{length}/{metric}': float(np.mean(values))
                for (group, length, metric), values in buckets.items()}

    trace=(args.output/'sample_trace.jsonl').open('a' if args.resume else 'w')
    legacy_starts={(r['sequence_id'],int(L)):set(starts) for r in map(json.loads,(SOURCE/'intervals.jsonl').read_text().splitlines()) if r['dataset']=='trumans' for L,starts in r['valid_starts_30fps'].items()}
    model.train()
    for step in range(first + 1, args.steps + 1):
        phase = set_training_phase(model,step,args.encoder_only_steps)
        group, length = schedule[step - 1]
        batch,identities=sample_batch(train,length,group,args.batch_size,return_identities=True)
        batch=to_cuda(batch)
        clean = batch['motion']
        timestep = torch.randint(len(alphas), (len(clean),), device='cuda')
        if args.low_noise_fraction:
            low_noise = torch.rand(len(clean), device='cuda') < args.low_noise_fraction
            low_t = torch.randint(100, (len(clean),), device='cuda')
            timestep = torch.where(low_noise, low_t, timestep)
        alpha = alphas[timestep][:, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
        control_mask, control_modes = sample_control_masks(
            len(clean), length, condition_rng, clean.device)
        scene_draw = condition_rng.random()
        use_scene = bool(not args.disable_scene and scene_draw >= args.scene_dropout)
        batch = prepare_observation(batch,args.observation_protocol,generator=observation_rng,step=step,seed=args.seed)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            predicted = model(noisy, timestep, batch, control_mask=control_mask,
                              use_scene=use_scene)
        predicted=apply_head_constraint(predicted.float(),batch,control_mask,group)
        losses = supervised_losses(predicted.float(), clean, batch,
                                   profile=args.loss_profile, signal_weight=alpha.flatten())
        if args.rich_contact_root:
            from experiments.offline_camera_retrain_v1.contact_supervision import contact_loss,contact_metrics
            logits=model.contact_logits(predicted.float(),batch,use_scene=use_scene)
            losses['contact_bce']=contact_loss(logits,batch,alpha.flatten())
            losses.update(contact_metrics(logits,batch))
            losses['total']=losses['total']+args.contact_loss_weight*losses['contact_bce']
        loss = losses['total']
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        update_learning_rates(optimizer,step,args.steps,args.lr,args.warmup_steps,args.encoder_only_steps)
        optimizer.step()
        trace.write(json.dumps(dict(step=step,group=group,length=length,pairing=dict(noisy_sha256=hashlib.sha256(noisy.detach().cpu().numpy().tobytes()).hexdigest(),timestep=timestep.tolist(),control_sha256=hashlib.sha256(control_mask.cpu().numpy().tobytes()).hexdigest(),scene=use_scene),samples=[dict(sequence_id=i['sequence_id'],source_start_30fps=i['source_start_30fps'],outside_legacy_static_windows=(i['source_start_30fps'] not in legacy_starts.get((i['sequence_id'],length),set())) if group=='trumans' else None) for i in identities]))+'\n')
        if step%20==0:trace.flush()
        if step == 1 or step % 20 == 0:
            row = dict(step=step, phase=phase,encoder_lr=optimizer.param_groups[0]['lr'],body_lr=optimizer.param_groups[1]['lr'], length=length, group=group, elapsed_s=time.time() - start_time,
                       control_modes=control_modes,
                       scene=use_scene,
                       lr=optimizer.param_groups[0]['lr'],
                       gradient_norm=float(gradient_norm),
                       samples_seen=step * args.batch_size,
                       peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                       losses={key: float(value.detach()) for key, value in losses.items()})
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
        if step % args.save_every == 0 or step == args.steps:
            metrics = validation_metrics()
            score = float(np.mean([v for k, v in metrics.items() if k.endswith('/fk_mpjpe_cm')]))
            row = dict(step=step, validation=metrics, validation_mean_fk_mpjpe_cm=score)
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            improved = score < best
            best = min(best, score)
            checkpoint = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                              config=config, best_validation=best,
                              torch_rng_state=torch.get_rng_state(),
                              cuda_rng_state=torch.cuda.get_rng_state(),
                              numpy_rng_state=np.random.get_state(),
                              data_rng_state=train.base.rng.bit_generator.state,
                              sampler_state=train.sampler_state(),
                              condition_rng_state=condition_rng.bit_generator.state,
                              training_phase=phase,
                              observation_rng_state=observation_rng.get_state())
            save_checkpoint(checkpoint,args.output/'last.pt')
            if improved:
                save_checkpoint(checkpoint,args.output/'best.pt')
            if args.ddim_every and (step % args.ddim_every == 0 or step == args.steps):
                evaluation_dir = args.output / 'evaluations' / f'step_{step:06d}'
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-u', '-m',
                           'experiments.offline_camera_retrain_v1.evaluate_sampling',
                           '--checkpoint', str((args.output / 'last.pt').resolve()),
                           '--output', str((evaluation_dir / 'metrics.json').resolve()),
                           '--gallery', str((evaluation_dir / 'gallery.html').resolve()),
                           '--length', '128', '--ddim-steps', '20', '--samples-per-group', '4']
                if args.ddim_head_only:
                    command.append('--head-only')
                with (evaluation_dir / 'evaluation.log').open('w') as evaluation_log:
                    result = subprocess.run(command, stdout=evaluation_log,
                                            stderr=subprocess.STDOUT, check=False)
                row = dict(step=step, sampling_exit_code=result.returncode,
                           evaluation_dir=str(evaluation_dir))
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + '\n')
                if result.returncode:
                    raise RuntimeError(f'Sampling failed; inspect {evaluation_dir}/evaluation.log')
                if step % args.keep_inference_every == 0 or step == args.steps:
                    save_checkpoint(dict(step=step, model=model.state_dict(), config=config),
                               evaluation_dir / 'inference.pt')
                write_run_index(args.output)
    trace.close()
    log.close()


if __name__ == '__main__':
    main()
