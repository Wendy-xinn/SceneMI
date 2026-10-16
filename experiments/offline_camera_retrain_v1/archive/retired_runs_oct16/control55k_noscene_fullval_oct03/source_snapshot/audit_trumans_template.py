"""Verify released TRUMANS joints against male versus neutral SMPL-X templates.

Does not overwrite any archived pose/rest arrays. The optional exported
template is used only with an explicitly versioned skeleton profile.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from smplx.lbs import vertices2joints
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model, numpy_fk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-rest', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    models = {g: load_model('smplx', g) for g in ('neutral', 'male')}
    rests = {g: vertices2joints(m.J_regressor, m.v_template[None])[0, :22].detach().numpy()
             for g, m in models.items()}
    root = Path(__file__).parent / 'data/poses'
    rows = []
    for split in ('train', 'validation', 'test'):
        for item in map(json.loads, (root / f'{split}.jsonl').read_text().splitlines()):
            if item['dataset'] != 'trumans':
                continue
            folder = Path(item['folder'])
            pose = np.load(folder / 'pose_axis_angle_world.npy', mmap_mode='r')
            trans = np.load(folder / 'translation_world.npy', mmap_mode='r')
            truth = np.load(Path(item['prepared_folder']) / 'joints_world.npy', mmap_mode='r')
            for frame in sorted({0, 2 * (len(pose) // 4), 2 * ((len(pose)-1) // 2)}):
                values = {}
                for gender, rest in rests.items():
                    joints = numpy_fk(pose[frame], trans[frame], rest, models[gender].parents[:22])
                    error = np.linalg.norm(joints - truth[frame], axis=-1)
                    aligned = joints + truth[frame, 0] - joints[0]
                    aligned_error = np.linalg.norm(aligned - truth[frame], axis=-1)
                    values[gender] = dict(mpjpe_m=float(error.mean()), foot_error_m=float(error[[10, 11]].mean()),
                                          max_joint_error_m=float(error.max()),
                                          root_aligned_max_joint_error_m=float(aligned_error.max()),
                                          root_aligned_mpjpe_m=float(aligned_error.mean()))
                rows.append(dict(split=split, sequence_id=item['sequence_id'], frame=frame, metrics=values))
        print('audited', split, flush=True)
    summary = {}
    for split in ('train', 'validation', 'test'):
        selected = [r for r in rows if r['split'] == split]
        summary[split] = {gender: dict(
            n=len(selected), mpjpe_m=float(np.mean([r['metrics'][gender]['mpjpe_m'] for r in selected])),
            foot_error_m=float(np.mean([r['metrics'][gender]['foot_error_m'] for r in selected])),
            max_joint_error_m=max(r['metrics'][gender]['max_joint_error_m'] for r in selected),
            root_aligned_max_joint_error_m=max(r['metrics'][gender]['root_aligned_max_joint_error_m'] for r in selected))
            for gender in rests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, cases=rows), indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    # Corrected-profile root translation is taken from the annotated pelvis.
    # One released training sequence has a ~1.7 cm global translation bias;
    # retain the absolute audit above instead of hiding that discrepancy.
    if any(s['male']['root_aligned_max_joint_error_m'] > .005 for s in summary.values()):
        raise RuntimeError('Male template failed strict 5 mm maximum-joint audit; do not export')
    if args.export_rest:
        if args.export_rest.exists():
            raise FileExistsError(args.export_rest)
        args.export_rest.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.export_rest, rests['male'].astype(np.float32))


if __name__ == '__main__':
    main()
