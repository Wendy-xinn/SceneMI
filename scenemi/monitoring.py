"""Training progress inspection and terminal/TensorBoard reporting.

Read-only helpers do not import torch or touch checkpoints.
"""
import hashlib
import json
from pathlib import Path
import time


class TrainingReporter:
    """Report completed updates; validation time is excluded from train speed."""

    def __init__(self, output, config, initial_step=0):
        from tqdm.auto import tqdm
        self.writer = None
        self.progress = None
        self.tqdm = tqdm
        self.compute_seconds = 0.
        self.updates = 0
        if config.get('logging', {}).get('tensorboard', True):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError('TensorBoard logging requires tensorboard') from error
            self.writer = SummaryWriter(str(Path(output) / 'tensorboard'),
                                        purge_step=initial_step + 1 if initial_step else None)
            self.writer.add_text('run/config', json.dumps(config, ensure_ascii=False), initial_step)
        label = 'EGO + scene' if config.get('version') == 'ego-scene-adaptation-v1' else 'EGO-only · 4 datasets'
        self.progress = tqdm(total=config['training']['planned_updates'], initial=initial_step,
                             desc=label, unit='step', dynamic_ncols=True)

    def update(self, record, seconds):
        step = record['step']
        self.compute_seconds += seconds
        self.updates += 1
        per_step = self.compute_seconds / self.updates
        eta = max(0, self.progress.total - step) * per_step
        self.progress.update(1)
        self.progress.set_postfix(loss=f"{record['total']:.4f}", lr=f"{record['lr']:.2e}",
                                  sec=f'{per_step:.2f}', ETA=f'{eta / 3600:.2f}h')
        if self.writer is not None:
            for key, value in {'loss_total': record['total'], 'gradient_norm': record['gradient_norm'],
                               'learning_rate': record['lr'], 'seconds_per_step': seconds}.items():
                self.writer.add_scalar('train/' + key, value, step)
            for key, value in record.get('terms', {}).items():
                self.writer.add_scalar('train/loss/' + key, value, step)
            for key in ('camera_noise_fraction', 'camera_noise_strength',
                        'grail_contact_weight_mean', 'grail_contact_weight_zero_fraction'):
                if key in record:
                    self.writer.add_scalar('train/' + key, record[key], step)

    def validation(self, result, step, prefix):
        metrics = ' | '.join(f"{name} {values['mpjpe_mm']:.2f}" for name, values in result['by_dataset'].items())
        self.tqdm.write(f"{prefix} step {step}: MPJPE {result['selection_score_mm']:.2f} mm | {metrics}")
        if self.writer is not None:
            self.writer.add_scalar(prefix + '/selection_score_mm', result['selection_score_mm'], step)
            for name, values in dict(result['by_dataset'], macro=result['dataset_macro_mean']).items():
                for key, value in values.items():
                    self.writer.add_scalar(f'{prefix}/{name}/{key}', value, step)
            self.writer.flush()

    def best(self, best):
        if best is not None:
            label = 'clean/noisy mean' if 'noisy_score_mm' in best else 'MPJPE'
            self.tqdm.write(f"Best: step {best['step']} | {label} {best['selection_score_mm']:.2f} mm | {best['checkpoint']}")

    def close(self):
        if self.progress is not None:
            self.progress.close()
        if self.writer is not None:
            self.writer.close()


def progress_snapshot(run):
    path = Path(run)/'training/updates.jsonl'
    if not path.exists():
        return None
    with path.open('rb') as stream:
        data = stream.read()
    # The writer may be between write() and flush(). A partial last record
    # does not represent a completed update.
    data = data[:data.rfind(b'\n')+1]
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    steps = [int(row['step']) for row in rows]
    return dict(path=str(path), logged_updates=len(rows),
                first_update=steps[0] if steps else None,
                last_update=steps[-1] if steps else None,
                contiguous=all(b == a+1 for a, b in zip(steps, steps[1:])),
                latest=rows[-1] if rows else None,
                _bytes=len(data), _sha256=hashlib.sha256(data).hexdigest(),
                _monotonic=time.monotonic())


def public_progress(snapshot):
    if snapshot is None:
        return None
    return {k:v for k,v in snapshot.items() if not k.startswith('_')}


def measure_progress(before, after, planned_updates=None):
    """Use only new updates and monotonic elapsed time; reject rewritten logs."""
    result = dict(updates_per_second=None, seconds_per_update=None,
                  remaining_seconds=None)
    if before is None or after is None:
        return dict(result, state='log_unavailable')
    elapsed = after['_monotonic']-before['_monotonic']
    result['elapsed_seconds'] = elapsed
    # A recovery may truncate and replay a log within this observation interval.
    # Count neither duplicated updates nor a change of attempt as throughput.
    with Path(after['path']).open('rb') as stream:
        prefix = stream.read(before['_bytes'])
    if (not before['contiguous'] or not after['contiguous']
            or after['_bytes'] < before['_bytes']
            or hashlib.sha256(prefix).hexdigest() != before['_sha256']):
        return dict(result, state='log_rewritten_or_noncontiguous')
    updates = after['logged_updates']-before['logged_updates']
    result['new_updates'] = updates
    if elapsed <= 0 or updates < 0:
        return dict(result, state='invalid_observation_interval')
    if not updates:
        return dict(result, state='no_updates_observed')
    result.update(state='advancing', updates_per_second=updates/elapsed,
                  seconds_per_update=elapsed/updates)
    if planned_updates is not None and after['last_update'] is not None:
        result['remaining_seconds'] = max(0, planned_updates-after['last_update'])*elapsed/updates
    return result
