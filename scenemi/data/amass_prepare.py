"""Resumable, CPU-only execution and auditing of pinned EgoAllo stage 0a.

The NPZ files are ground-truth records, not a promise that every field is an
inference input. Training input policy belongs to the subsequent data adapter.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from scenemi.config import ROOT
from scenemi.data.amass import SPLIT_FILE, exclusion_reason, verify_upstream
from scenemi.geometry.body import AMASS_ZUP_TO_YUP, PARENTS, encode_body, smplh_profile
from scenemi.geometry.camera import camera_from_head, cpf_from_camera, head_from_camera, transform

PROTOCOL = 'egoallo-0a-cpu-audited-v1'
GIB = 2 ** 30


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def source_indices(frames, fps):
    """Clock of the BODY fields of 0a; the consumer then drops first frame.

    Upstream does not edge-trim its hand/constant-beta/alignment arrays in the
    same way. Preserve its NPZ verbatim; do not treat these as body frame IDs.
    """
    start, end = int(.1 * frames), int(.9 * frames)
    if end - start < fps:
        return np.empty(0, dtype=np.int64)
    trimmed = end - start - 2
    return start + 1 + np.linspace(0, trimmed - 1, num=int(30 / fps * trimmed), dtype=np.int64)


def inspect_raw(path):
    """Reject unexpected schemas/nonfinite targets rather than silently skip."""
    with np.load(path, allow_pickle=False) as data:
        poses, trans, betas = data['poses'], data['trans'], data['betas']
        fps = float(data['mocap_framerate'])
        gender = data['gender'].item()
        gender = gender.decode('utf-8') if isinstance(gender, bytes) else str(gender)
        if poses.ndim != 2 or poses.shape[1] != 156 or trans.shape != (len(poses), 3):
            raise ValueError(f'Unexpected SMPL-H pose/translation schema: {path}')
        if betas.shape != (16,) or gender not in ('male', 'female'):
            raise ValueError(f'Unexpected native shape/gender: {path}')
        if not all(np.isfinite(v).all() for v in (poses, trans, betas)) or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Nonfinite or invalid raw motion: {path}')
        frames = len(poses)
    # Exactly the two correction rules from the unchanged upstream loader.
    if 'BMLhandball' in str(path):
        fps = 240.
    if '20160930_50032' in str(path) or '20161014_50033' in str(path):
        fps = 59.
    return dict(raw_frames=frames, source_fps=fps, raw_gender=gender,
                body_source_frame_ids=source_indices(frames, fps))


def audit_processed(path, raw_info, model_path):
    """Independent scipy/numpy FK, shape and SE(3) checks against upstream."""
    ids = raw_info['body_source_frame_ids']
    with np.load(path, allow_pickle=False) as data:
        n = int(data['num_frames'])
        if n != len(ids) or int(data['fps']) != 30 or str(data['gender']) != 'neutral':
            raise ValueError('Processed metadata/clock differs from stage 0a')
        for name in data.files:
            value = data[name]
            if value.dtype.kind in 'fci' and not np.isfinite(value).all():
                raise ValueError(f'Nonfinite processed field: {name}')
        expected = {'root_orient': (n, 3), 'pose_body': (n, 63), 'trans': (n, 3),
                    'joints': (n, 22, 3), 'betas': (n, 16), 'contacts': (n, 22)}
        for name, shape in expected.items():
            if data[name].shape != shape:
                raise ValueError(f'Unexpected processed {name} shape: {data[name].shape}')
        if not np.allclose(data['betas'], data['betas'][:1], atol=1e-6):
            raise ValueError('Body shape is not constant within a motion')
        profile = smplh_profile(model_path, data['betas'][0])
        local = Rotation.from_rotvec(np.concatenate((data['root_orient'], data['pose_body']), -1)
                                     .reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
        local[:, 0] = AMASS_ZUP_TO_YUP @ local[:, 0]
        truth = data['joints'].astype(np.float64) @ AMASS_ZUP_TO_YUP.T
        motion = encode_body(local, truth[:, 0], profile['rest'])
        recovered = motion[:, 135:].reshape(n, 22, 3) * 2
        fk_error = float(np.linalg.norm(recovered - truth, axis=-1).max())
        if fk_error > 2e-4:
            raise ValueError(f'Independent FK does not match upstream joints: {fk_error} m')
        globals_ = [local[:, 0]]
        for joint in range(1, 22):
            globals_.append(globals_[PARENTS[joint]] @ local[:, joint])
        head = transform(globals_[15], truth[:, 15])
        camera = camera_from_head(head, profile['head_camera'])
        head_error = float(np.abs(head_from_camera(camera, profile['head_camera']) - head).max())
        cpf_error = float(np.abs(cpf_from_camera(camera, profile['head_camera'], profile['head_cpf'])
                                 - head @ profile['head_cpf']).max())
        if max(head_error, cpf_error) > 1e-10:
            raise ValueError('Camera/head/CPF round trip failed')
        # Consumer alignment with EgoAllo 0b: retain only frames [1:].
        usable = max(n - 1, 0)
        return dict(frames_0a=n, frames_after_relative_pose_drop=usable,
                    eligible_128=usable >= 128, possible_windows_128=max(0, usable - 127),
                    fk_max_error_m=fk_error, head_roundtrip_error=head_error,
                    cpf_roundtrip_error=cpf_error,
                    source_first_frame=int(ids[1]) if usable else None,
                    source_last_frame=int(ids[-1]) if usable else None,
                    source_body_frame_ids_sha256=hashlib.sha256(ids[1:].tobytes()).hexdigest(),
                    bytes=Path(path).stat().st_size, sha256=sha256(path))


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_preprocessor(repository):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise ValueError('This preparation command requires an explicitly CPU-only child')
    sys.path.insert(0, str(repository / 'src'))
    spec = importlib.util.spec_from_file_location('_scenemi_egoallo_0a', repository / '0a_preprocess_training_data.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.guru.remove()
    module.guru.add(sys.stderr, level='WARNING')
    return module


def seal_dependencies(paths, inventory_path, module):
    """Seal the loaded preprocessing closure, not the entire legacy tree."""
    upstream = verify_upstream(paths['egoallo_repo'])
    for imported in list(sys.modules.values()):
        name = getattr(imported, '__name__', '')
        file = getattr(imported, '__file__', None)
        if name.startswith('egoallo.') and file and file.endswith('.py'):
            file = Path(file).resolve()
            relative = file.relative_to(paths['egoallo_repo']).as_posix()
            upstream['source_sha256'][relative] = sha256(file)
            # Check dynamically imported files are also the pinned originals.
            import subprocess
            original = subprocess.check_output(['git', '-C', str(paths['egoallo_repo']), 'show',
                                                f"{upstream['commit']}:{relative}"])
            if hashlib.sha256(original).hexdigest() != upstream['source_sha256'][relative]:
                raise ValueError(f'Locally modified preprocessing dependency: {relative}')
    local_names = ['scenemi/data/amass_prepare.py', 'scenemi/data/amass.py',
                   'scenemi/geometry/body.py', 'scenemi/geometry/camera.py', 'configs/amass_splits.json']
    assets = [paths['smplh_models'] / 'neutral/model.npz'] + [
        paths['egoallo_repo'] / f'data/smplh_gender_conversion/{gender}_to_neutral.npz'
        for gender in ('male', 'female')]
    packages = ('numpy', 'scipy', 'torch', 'smplx', 'scikit-learn', 'tyro', 'jaxtyping', 'loguru', 'matplotlib', 'einops')
    return dict(protocol=PROTOCOL, upstream=upstream, python=sys.version,
                packages={name: importlib.metadata.version(name) for name in packages},
                local_source_sha256={name: sha256(ROOT / name) for name in local_names},
                assets_sha256={str(path): sha256(path) for path in assets},
                inventory_sha256=sha256(inventory_path),
                raw_root=str(paths['amass_raw']), output_root=str(paths['prepared_amass']),
                parameters=dict(beta_neutral=True, reflect=[False, True], out_fps=30,
                                split_frame_limit=256, save_velocities=True, save_verts=False),
                annotation='Ground truth shape and mounts are labels; inference input policy is not decided by this export.')


def verified_records(ledger, output_root, inventory_rows):
    records = {}
    if not ledger.exists():
        return records
    known = {row['source']: row for row in inventory_rows}
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        source = row['source']
        if source in records or source not in known:
            raise ValueError('Duplicate or unknown manifest identity')
        for key in ('bytes', 'mtime_ns', 'split', 'family'):
            if row['inventory'][key] != known[source][key]:
                raise ValueError('Resume manifest differs from the raw inventory')
        for variant in row.get('variants', []):
            target = output_root / variant['path']
            if not target.is_file() or sha256(target) != variant['sha256']:
                raise ValueError(f'Resume output changed or missing: {target}')
        records[source] = row
    return records


def prepare(paths, inventory_path, *, pilot=False):
    """Pilot one eligible motion per corpus, or prepare all inventoried motions.

    Both modes share outputs and a verified manifest. Uncommitted partial
    products are recomputed on resume; committed products must match SHA256.
    """
    import fcntl
    import torch
    inventory_path = Path(inventory_path).resolve()
    output = paths['prepared_amass']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    module = load_preprocessor(paths['egoallo_repo'])
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    job = seal_dependencies(paths, inventory_path, module)
    stamp = output / 'JOB.json'
    if stamp.exists():
        if json.loads(stamp.read_text()) != job:
            raise ValueError('Preparation dependencies changed; use a new output root')
    else:
        if any(path.name != '.prepare.lock' for path in output.iterdir()):
            raise ValueError('Refuse to adopt an output directory without a provenance seal')
        atomic_json(stamp, job)
    rows = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    summary = json.loads(inventory_path.with_suffix('.summary.json').read_text())
    if summary['inventory_sha256'] != job['inventory_sha256']:
        raise ValueError('Inventory hash differs from its original audit')
    ledger = output / 'manifest.jsonl'
    records = verified_records(ledger, output, rows)
    status_path = paths['output'] / 'preparation/STATUS.json'
    status_path.parent.mkdir(parents=True, exist_ok=True)
    requested = {row['corpus'] for row in rows}
    covered = {row['corpus'] for row in records.values() if row.get('eligible_128')}
    started = time.monotonic()
    finished_this_call = 0
    source = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous_handler = signal.signal(signal.SIGTERM, stop)

    def report(state, error=None):
        variants = [v for row in records.values() for v in row.get('variants', [])]
        counts = Counter(row['state'] for row in records.values())
        result = dict(protocol=PROTOCOL, state=state, pilot=pilot, pid=os.getpid(),
                      updated_at_utc=datetime.now(timezone.utc).isoformat(),
                      total_raw_files=len(rows), committed_sources=len(records), counts=dict(counts),
                      completed_variants=len(variants), eligible_corpora=sorted(covered),
                      eligible_128_variants=sum(v['eligible_128'] for v in variants),
                      fk_max_error_m=max((v['fk_max_error_m'] for v in variants), default=None),
                      output_bytes=sum(v['bytes'] for v in variants),
                      manifest=str(ledger), output_root=str(output), current_source=source,
                      invocation_elapsed_seconds=time.monotonic() - started,
                      invocation_completed_sources=finished_this_call, failures=int(error is not None),
                      error=error, optimizer_steps=0, no_gpu=True, body_shape_is_label=True)
        atomic_json(status_path, result)
        return result

    try:
        report('running_pilot' if pilot else 'running')
        for row in rows:
            source = row['source']
            if source in records or (pilot and row['corpus'] in covered):
                continue
            if stopping:
                return report('stopped_at_sequence_boundary')
            raw = paths['amass_raw'] / source
            if (raw.stat().st_size, raw.stat().st_mtime_ns) != (row['bytes'], row['mtime_ns']):
                raise ValueError(f'Raw file changed since inventory: {source}')
            if exclusion_reason(source) != row['exclusion']:
                raise ValueError('Exclusion rule differs from inventory')
            record = dict(source=source, family=row['family'], split=row['split'], corpus=row['corpus'],
                          inventory={key: row[key] for key in ('bytes', 'mtime_ns', 'split', 'family')},
                          variants=[], eligible_128=False)
            if row['exclusion']:
                record.update(state='filename_excluded', reason=row['exclusion'])
            else:
                info = inspect_raw(raw)
                record.update(raw_frames=info['raw_frames'], source_fps=info['source_fps'],
                              raw_gender=info['raw_gender'], raw_sha256=sha256(raw))
                if not len(info['body_source_frame_ids']):
                    record.update(state='too_short_upstream', reason='middle_80_percent_shorter_than_one_second')
                else:
                    if shutil.disk_usage(output).free < 20 * GIB:
                        raise RuntimeError('Less than 20 GiB free on output volume; stopped before next sequence')
                    for reflect in (False, True):
                        relative = Path('neutral') / source
                        if reflect:
                            relative = relative.with_name(relative.stem + '_reflect.npz')
                        target = output / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name(target.stem + '.partial.npz')
                        # Only this sealed job owns these uncommitted products.
                        temporary.unlink(missing_ok=True)
                        with working_directory(paths['egoallo_repo']):
                            module.process_seq(str(raw), str(temporary), str(paths['smplh_models']),
                                               dev_id=0, beta_neutral=True, reflect=reflect,
                                               split_frame_limit=256, overwrite=False)
                        if not temporary.is_file():
                            raise ValueError('Upstream unexpectedly discarded an eligible raw motion')
                        audit = audit_processed(temporary, info, paths['smplh_models'] / 'neutral/model.npz')
                        temporary.replace(target)
                        record['variants'].append(dict(path=relative.as_posix(), reflected=reflect, **audit))
                    record.update(state='prepared', eligible_128=all(v['eligible_128'] for v in record['variants']))
                    if record['eligible_128']:
                        covered.add(row['corpus'])
            with ledger.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            records[source] = record
            finished_this_call += 1
            report('running_pilot' if pilot else 'running')
            if record['state'] == 'prepared':
                print(json.dumps(dict(source=source, split=row['split'], eligible_128=record['eligible_128'],
                                      committed=len(records), elapsed=round(time.monotonic()-started, 2))), flush=True)
            if pilot and covered == requested:
                break
        if pilot and covered != requested:
            raise ValueError(f'No eligible pilot motion found for corpora: {sorted(requested-covered)}')
        result = report('pilot_complete' if pilot else 'complete')
        audit_path = status_path.with_name('PILOT.json' if pilot else 'COMPLETE.json')
        atomic_json(audit_path, result)
        return result
    except Exception as error:
        report('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
