"""Legacy 5 Hz inspection export, retained to reproduce historical 55k runs.

Production native_v1 training uses export_all_native20_training_scenes.py,
which decodes raw 30 Hz fits directly on the shared 20 Hz scene/motion clock.
This script's stride=6 outputs must not be upsampled into that protocol.
"""
import argparse, csv, json, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
rich = Path('/home/wenxin/projects/RICH/extracted')
python = Path('/home/wenxin/miniconda3/envs/scenemi/bin/python')
script = HERE / 'prepare_rich_pseudo_ego.py'
parser = argparse.ArgumentParser()
parser.add_argument('--output-suffix', default='faceout_oct07')
parser.add_argument('--negative-z-diagnostic', action='store_true',
                    help='Diagnostic only: points optical +Z into the fitted head')
args = parser.parse_args()
log = Path(f'/tmp/rich_preprocess_timing_{args.output_suffix}.csv')
rows = []
for split in ('train', 'val'):
    root = rich / f'{split}_body'
    out = HERE / 'data/scene_visibility_v2_oct05' / f'rich_{split}_{args.output_suffix}'
    out.mkdir(parents=True, exist_ok=True)
    for seqdir in sorted(p for p in root.iterdir() if p.is_dir()):
        seq = seqdir.name
        plys = sorted(seqdir.glob('*/*.ply'), key=lambda p: (int(p.parent.name), p.name))
        if not plys:
            continue
        subject = plys[0].stem
        target = out / seq / 'metadata.json'
        if target.is_file():
            meta = json.loads(target.read_text())
            protocol = ('SMPL-X -Z optical forward' if args.negative_z_diagnostic
                        else 'identity face mount')
            if protocol not in meta.get('camera_protocol', ''):
                raise ValueError(f'{target}: existing output has the wrong camera protocol')
            rows.append(dict(split=split, sequence=seq, subject=subject, status='skipped', seconds=0.0))
            continue
        start = time.monotonic()
        cmd = [str(python), '-u', str(script), '--sequence', seq,
               '--split', split, '--subject-id', subject, '--stride', '6', '--max-frames', '0',
               '--output', str(out)]
        if args.negative_z_diagnostic:
            cmd.append('--negative-z-forward')
        result = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        seconds = time.monotonic() - start
        (out / f'{seq}.log').write_text(result.stdout)
        status = 'ok' if result.returncode == 0 and target.is_file() else f'fail:{result.returncode}'
        rows.append(dict(split=split, sequence=seq, subject=subject,
                         status=status, seconds=round(seconds, 2)))
        with log.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['split','sequence','subject','status','seconds'])
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(rows[-1]), flush=True)
print(json.dumps({'completed': len(rows), 'log': str(log)}), flush=True)
