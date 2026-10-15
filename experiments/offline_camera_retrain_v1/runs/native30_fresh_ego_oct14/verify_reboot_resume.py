"""Verify resumed sampling against the archived, unsaved pre-crash updates."""
import datetime
import argparse
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main(archive=ROOT / 'reboot_oct15'):
    saved = json.loads((archive / 'RECOVERY.json').read_text())['checkpoint_step']
    previous = []
    for line in (archive / f'trace_after_saved_{saved}.raw').read_bytes().splitlines():
        try:
            previous.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            break
    wanted = {x['step']: x for x in previous[:5]}
    began = time.monotonic()
    while time.monotonic() - began < 1800:
        rows = []
        for line in (ROOT / 'training/trace.jsonl').read_bytes().splitlines():
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
        if rows[-1]['step'] >= max(wanted):
            assert [x['step'] for x in rows] == list(range(1, rows[-1]['step'] + 1))
            comparisons = []
            for row in rows:
                if row['step'] not in wanted:
                    continue
                old = wanted[row['step']]
                keys = ('group', 'mode', 'sequence_id', 'start', 'yaw_augmentation_rad',
                        'source_start_30fps', 'anchor', 'origin', 'sampling')
                identities = lambda x: [{k: identity[k] for k in keys} for identity in x['identities']]
                checks = dict(timesteps_equal=row['timesteps'] == old['timesteps'],
                    scene_dropout_equal=row['scene'] == old['scene'], lr_equal=row['lr'] == old['lr'],
                    sampled_identities_and_yaw_equal=identities(row) == identities(old),
                    finite=math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']))
                comparisons.append(dict(step=row['step'], checks=checks,
                    loss_before_crash=old['loss'], loss_after_resume=row['loss'],
                    absolute_loss_difference=abs(row['loss'] - old['loss'])))
            report = dict(status='passed' if all(all(x['checks'].values()) for x in comparisons) else 'failed',
                observed_at=datetime.datetime.now().astimezone().isoformat(), resumed_from_step=saved,
                latest_step=rows[-1]['step'], wait_s=time.monotonic()-began, comparisons=comparisons,
                numerical_note='Small loss/gradient differences can occur across CUDA restarts; report rather than claim bitwise continuation.')
            (archive / 'RESUME_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
            assert report['status'] == 'passed'
            return
        time.sleep(15)
    raise TimeoutError('No five resumed updates within 30 minutes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'reboot_oct15')
    main(parser.parse_args().archive)
