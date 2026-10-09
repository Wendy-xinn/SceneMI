"""Cache body-shape rest joints for differentiable SMPL(-X) forward kinematics."""
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
for name, value in dict(bool=bool, int=int, float=float, complex=complex,
                        object=object, unicode=str, str=str).items():
    if name not in np.__dict__:
        setattr(np, name, value)
import smplx
from smplx.lbs import blend_shapes, vertices2joints

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
MODELS = PROJECTS / 'SceneMI/body_models'
EGOBODY = PROJECTS / 'egobody'


def load_model(model_type, gender):
    if model_type == 'smplx':
        return smplx.create(str(MODELS), model_type='smplx', gender=gender,
                            num_betas=10, use_pca=False, batch_size=1)
    if model_type == 'smpl':
        path = PROJECTS / 'ProtoMotions/data/smpl' / f'SMPL_{gender.upper()}.pkl'
        return smplx.SMPL(str(path), batch_size=1)
    raise ValueError(model_type)


def numpy_fk(pose, translation, rest, parents):
    local = Rotation.from_rotvec(pose).as_matrix()
    global_rot = np.empty_like(local)
    joints = np.empty((len(rest), 3), np.float32)
    global_rot[0] = local[0]
    joints[0] = rest[0] + translation
    for joint in range(1, len(rest)):
        parent = int(parents[joint])
        global_rot[joint] = global_rot[parent] @ local[joint]
        joints[joint] = joints[parent] + global_rot[parent] @ (rest[joint] - rest[parent])
    return joints


def main():
    models = {}
    summary = {}
    for split in ('train', 'validation', 'test'):
        manifest = HERE / 'data/poses' / f'{split}.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        errors = {'trumans': [], 'egobody': []}
        for row in rows:
            key = (row['model'], row['gender'])
            if key not in models:
                models[key] = load_model(*key)
            model = models[key]
            shape = torch.tensor([row['betas']], dtype=torch.float32)
            with torch.no_grad():
                shaped = model.v_template[None] + blend_shapes(shape, model.shapedirs[:, :, :10])
                rest = vertices2joints(model.J_regressor, shaped)[0, :22].numpy()
            folder = Path(row['folder'])
            np.save(folder / 'rest_joints.npy', rest.astype(np.float32))
            pose = np.load(folder / 'pose_axis_angle_world.npy', mmap_mode='r')
            translation = np.load(folder / 'translation_world.npy', mmap_mode='r')
            if row['model'] == 'smpl' and not row.get('root_rest_correction_applied', False):
                prepared_meta = json.loads((Path(row['prepared_folder']) / 'metadata.json').read_text())
                calibration = (EGOBODY / 'calibrations' / prepared_meta['recording'] /
                               'cal_trans/kinect12_to_world' / f'{prepared_meta["scene_id"]}.json')
                world_rotation = np.asarray(json.loads(calibration.read_text())['trans'])[:3, :3]
                translation = np.asarray(translation) + rest[0] @ world_rotation.T - rest[0]
                np.save(folder / 'translation_world.npy', translation.astype(np.float32))
                row['root_rest_correction_applied'] = True
                (folder / 'metadata.json').write_text(json.dumps(row, indent=2) + '\n')
            truth = np.load(Path(row['prepared_folder']) / 'joints_world.npy', mmap_mode='r')
            if not (len(pose) == len(translation) == len(truth) == row['frames_20fps']):
                raise ValueError(f'{row["sequence_id"]}: frame mismatch')
            selected = sorted(set((0, len(pose) // 2, len(pose) - 1)))
            for index in selected:
                predicted = numpy_fk(np.asarray(pose[index]), np.asarray(translation[index]),
                                     rest, model.parents[:22])
                errors[row['dataset']].append(float(np.linalg.norm(predicted - truth[index], axis=-1).mean()))
        summary[split] = {dataset: dict(cases=len(values), mean_m=float(np.mean(values)),
                                        p95_m=float(np.quantile(values, .95)),
                                        max_m=float(np.max(values)))
                          for dataset, values in errors.items()}
        print(split, json.dumps(summary[split]), flush=True)
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (HERE / 'data/poses/fk_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    if any(group['p95_m'] > .08 for split in summary.values() for group in split.values()):
        raise ValueError('FK supervision has excessive disagreement with prepared joints')


if __name__ == '__main__':
    main()
