"""Validate the AMASS-only prior before expensive preparation or training."""
from pathlib import Path
import math

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/amass128.yaml'


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get('version') != 'amass128-stage-a-v1':
        raise ValueError('Expected an amass128-stage-a-v1 configuration')
    data, model = config['data'], config['model']
    if data['fps'] != 30 or data['frames'] != 128 or data['source'] != 'amass':
        raise ValueError('Stage A requires AMASS at 30 Hz with 128-frame windows')
    splits = data['splits']
    if splits != {'train': ['train'], 'validation': ['val'], 'test': ['test']}:
        raise ValueError('Stage A keeps EgoAllo corpus train/val/test disjoint')
    if model['initialization'] != 'from_scratch' or model.get('checkpoint') is not None:
        raise ValueError('AMASS-only training must start from scratch without a checkpoint')
    if model['architecture'] != 'native30' or model['diffusion_target'] != 'absolute_x0':
        raise ValueError('Use the existing native30 architecture with absolute_x0')
    if model['latent_dim'] != 256 or model['dim_mults'] != [1, 2, 4]:
        raise ValueError('Preserve the existing backbone dimensions')
    if config['scene'] != {'enabled': False, 'geometry_losses': False}:
        raise ValueError('Stage A must disable scene conditions and geometry losses')
    observation = config['observation']
    if observation['ego_input'] != 'virtual_camera' or observation['ego_joints'] != [15]:
        raise ValueError('EGO uses the calibrated virtual camera without hand input')
    if observation.get('exo_input') != 'estimated_3d_body_or_joints':
        raise ValueError('EXO uses estimated 3D body/joints, with explicit availability and uncertainty')
    if observation.get('body_shape') != 'predicted_from_observations' or observation.get('ground_truth_shape_as_input') is not False:
        raise ValueError('Deployment has no ground-truth body shape or rest skeleton input')
    probabilities = observation['mode_probabilities']
    if set(probabilities) != {'ego', 'exo'} or any(v <= 0 for v in probabilities.values()):
        raise ValueError('Both EGO and EXO must receive training examples')
    if abs(sum(probabilities.values()) - 1) > 1e-8:
        raise ValueError('Mode probabilities must sum to one')
    if config['loss']['supervise_hidden_joints'] is not True:
        raise ValueError('Missing observations must still have full motion supervision')
    if data.get('representation') != '201d_common_neutral_pelvis_v1' or model.get('shape_betas') != 16:
        raise ValueError('Use a shape-independent root reference and all 16 native SMPL-H betas')
    if (data['body_model'] != 'smplh' or data['num_betas'] != 16 or
            data['sampling'] != 'uniform_family_then_variant_and_window' or
            data['exo_camera'] != 'unavailable_in_amass' or model['diffusion_steps'] != 1000):
        raise ValueError('Data/body/diffusion settings differ from the implemented AMASS protocol')
    if config['loss']['profile'] != 'native30_predicted_shape_v1':
        raise ValueError('Predicted and target shape require separate FK losses')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in config['loss']['coefficients'].values()):
        raise ValueError('Loss coefficients must be finite nonnegative numbers')
    if config['training']['precision'] != 'fp32' or config['training']['implementation'] != 'observation_shape_v1':
        raise ValueError('Use the validated fp32 observation-shape implementation')
    if observation['input_noise'] != 'disabled_clean_baseline':
        raise ValueError('Input noise must be calibrated and validated before enabling it')
    if observation['exo_mask'] != 'synthetic_body_part_temporal_dropout':
        raise ValueError('EXO stage A implements synthetic body/temporal missingness')
    from .data.observation_gaps import validate_policy
    validate_policy(observation['temporal_dropout'], data['frames'])
    from .resources import validate_runtime
    validate_runtime(config['runtime'])
    training, optimizer, validation = config['training'], config['optimizer'], config['validation']
    for key in ('batch_size', 'planned_updates', 'checkpoint_every', 'log_every'):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f'Training {key} must be a positive integer')
    if optimizer['name'] != 'AdamW' or training['checkpoint_policy'] != 'atomic_resumable_last':
        raise ValueError('Use AdamW with a single atomic resumable checkpoint')
    for key in ('lr', 'clip_norm'):
        if not math.isfinite(optimizer[key]) or optimizer[key] <= 0:
            raise ValueError(f'Optimizer {key} must be finite and positive')
    if (not math.isfinite(optimizer['weight_decay']) or optimizer['weight_decay'] < 0 or
            type(optimizer['warmup_updates']) is not int or optimizer['warmup_updates'] <= 0 or
            not 0 < optimizer['lr_floor_fraction'] <= 1):
        raise ValueError('Invalid AdamW warmup/decay schedule')
    if (validation['mode'] != ['ego', 'exo'] or validation['conditions'] != ['clean', 'temporal_gap']
            or validation['families_per_corpus'] != 1 or
            type(validation['every_updates']) is not int or validation['every_updates'] <= 0 or
            type(validation['ddim_steps']) is not int or not 2 <= validation['ddim_steps'] <= 1000 or
            not validation['fixed_seeds'] or
            any(type(seed) is not int for seed in validation['fixed_seeds'])):
        raise ValueError('Invalid fixed EGO/EXO development sampling protocol')
    return config


def load_paths(path):
    """Keep machine paths out of the publishable experiment configuration."""
    values = yaml.safe_load(Path(path).read_text())
    required = {'amass_raw', 'smplh_models', 'egoallo_repo', 'prepared_amass', 'output'}
    if set(values) != required:
        raise ValueError(f'Path file must contain exactly {sorted(required)}')
    return {key: Path(value).expanduser().resolve() for key, value in values.items()}


EGO_CONFIG = ROOT / 'configs/ego_only_4datasets.yaml'
EGO_SOURCES = ('amass', 'egobody', 'trumans', 'grail')


def load_ego_config(path=EGO_CONFIG):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text())
    c['_config_path'] = str(path)
    scene_stage = c['version'] == 'ego-scene-adaptation-v1'
    if c['version'] not in ('native-ego-four-corpus-scratch-v1', 'ego-camera-adaptation-v1', 'ego-scene-adaptation-v1'):
        raise ValueError('Wrong four-corpus EGO-only protocol')
    if c['data']['datasets'] != list(EGO_SOURCES) or c['data']['excluded_datasets'] != ['rich']:
        raise ValueError('The formal run must use AMASS/EgoBody/TRUMANS/GRAIL and exclude RICH')
    if c['validation']['test_used'] is not False:
        raise ValueError('Test data is forbidden during training and checkpoint selection')
    continuation = c.get('continuation')
    initialization = 'from_checkpoint' if continuation else 'from_scratch'
    if (c['model']['initialization'] != initialization or c['model']['scene_modules_trainable'] != scene_stage
            or c['data']['scene'] != ('observed_recording' if scene_stage else 'disabled')
            or c['data']['frames'] != 128 or c['data']['fps'] != 30
            or c['training']['precision'] != 'fp32'):
        raise ValueError('This trainer requires explicit initialization, scene-free 128-frame EGO in fp32')
    if continuation:
        if not scene_stage and (c['version'] != 'ego-camera-adaptation-v1' or not c.get('camera_noise')):
            raise ValueError('Continuation requires the declared camera-adaptation protocol')
        required = {'checkpoint', 'checkpoint_sha256', 'step', 'validation_manifest',
                    'validation_manifest_sha256', 'metrics', 'metrics_sha256'}
        if set(continuation) != required or type(continuation['step']) is not int or continuation['step'] <= 0:
            raise ValueError('Continuation requires a pinned checkpoint, step, manifest and metrics')
        for key in ('checkpoint_sha256', 'validation_manifest_sha256', 'metrics_sha256'):
            if len(continuation[key]) != 64 or any(x not in '0123456789abcdef' for x in continuation[key]):
                raise ValueError('Continuation identities must be SHA256 values')
        if c['training']['planned_updates'] <= continuation['step']:
            raise ValueError('Continuation budget must extend the saved step')
        if c['optimizer'].get('schedule') != ('scene_cosine' if scene_stage else 'constant'):
            raise ValueError('This adaptation uses an explicit constant learning rate without warmup restart')
        if not scene_stage:
            from scenemi.data.camera_noise import validate_policy
            validate_policy(c['camera_noise'])
        if c['validation'].get('primary') != ('guarded_scene_mpjpe_mm' if scene_stage else 'guarded_clean_noisy_mean_mpjpe_mm'):
            raise ValueError('Camera adaptation requires guarded clean/noisy selection')
        for key in ('clean_macro_regression_mm', 'clean_dataset_regression_mm'):
            if not math.isfinite(c['validation'][key]) or c['validation'][key] < 0:
                raise ValueError('Invalid clean validation regression guard')
    if scene_stage:
        if (not continuation or c.get('camera_noise') or c['scene_stage']['backbone'] != 'frozen'
                or c['scene_stage']['geometry_losses'] is not False
                or c['scene_stage']['contact_supervision'] != 'reliability_weighted_planting'
                or c['data']['history_voxel_m'] <= 0 or c['data']['frame_points'] <= 0):
            raise ValueError('Expected frozen EGO scene conditioning with private reliable contact labels')
        from scenemi.data.recording_scene import load_policy as map_policy
        from scenemi.contact_loss import load_policy as contact_policy
        observed = map_policy(ROOT / c['scene_stage']['map_policy'])
        contact_policy(ROOT / c['scene_stage']['contact_policy'])
        if c['local']['history_points'] != observed['local_point_limit']:
            raise ValueError('Scene history budget differs from map policy')
    for key in ('effective_batch_size', 'microbatch_size', 'planned_updates', 'checkpoint_every',
                'quick_validation_every'):
        if type(c['training'][key]) is not int or c['training'][key] <= 0:
            raise ValueError(f'Training {key} must be a positive integer')
    if c['data']['source_quota_per_update'] * 4 != c['training']['effective_batch_size']:
        raise ValueError('Source quotas must exactly fill the effective batch')
    if c['training']['effective_batch_size'] % c['training']['microbatch_size']:
        raise ValueError('Microbatch must divide the effective batch')
    if (type(c['model']['latent_dim']) is not int or c['model']['latent_dim'] <= 0
            or not c['model']['dim_mults']
            or any(type(v) is not int or v <= 0 for v in c['model']['dim_mults'])):
        raise ValueError('Invalid network dimensions')
    if c['optimizer']['warmup_updates'] <= 0 or not 0 < c['optimizer']['lr_floor_fraction'] <= 1:
        raise ValueError('Invalid warmup/learning-rate floor')
    runtime = c['runtime']
    allocator = runtime['allocator_gib']
    valid_allocator = allocator is None or (not isinstance(allocator, bool)
        and isinstance(allocator, (int, float)) and math.isfinite(allocator) and 0 < allocator <= 8)
    if (not 0 < runtime['memory_gib'] <= 16 or not valid_allocator
            or type(runtime['cpu_count']) is not int or not 1 <= runtime['cpu_count'] <= 4):
        raise ValueError('Runtime requires RAM <=16GiB, a positive allocator <=8GiB or null (uncapped), and <=4 CPU cores')
    for key in ('eval_batch_size', 'quick_families_per_dataset', 'ddim_steps'):
        if type(c['validation'][key]) is not int or c['validation'][key] <= 0:
            raise ValueError(f'Validation {key} must be a positive integer')
    return c


