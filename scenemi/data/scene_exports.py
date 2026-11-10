"""Read camera/body/scene exports as estimated inputs, without consulting GT.

Meshes can supply joint tracks without original SMPL pose/beta parameters.
The returned joint definition is mesh-regressed, not the native posed-joint
definition. World units and up direction stay explicitly uncalibrated.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import re
from zipfile import ZipFile

import numpy as np

from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.training import ensure_bounded, now


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np.core.numeric._frombuffer
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise ValueError(f'Unexpected pickle global: {module}.{name}')


def arrays(raw):
    return ArrayUnpickler(io.BytesIO(raw)).load()


def camera(value, source='hsi'):
    c = value['cameras'][source]
    c2w, w2c = np.asarray(c['camera_to_world_4x4']), np.asarray(c['world_to_camera_4x4'])
    if c2w.shape != (4, 4) or w2c.shape != (4, 4):
        raise ValueError('Expected homogeneous camera matrices')
    if not np.isfinite(c2w).all() or not np.isfinite(w2c).all():
        raise ValueError('Nonfinite estimated camera')
    np.testing.assert_allclose(c2w[3], [0, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(w2c[3], [0, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(c2w@w2c, np.eye(4), atol=3e-5)
    np.testing.assert_allclose(c2w[:3, :3].T@c2w[:3, :3], np.eye(3), atol=3e-5)
    if np.linalg.det(c2w[:3, :3]) < .999:
        raise ValueError('Camera reflection')
    intrinsic, size = np.asarray(value['intrinsic_3x3']), np.asarray(value['image_hw'])
    if (intrinsic.shape != (3, 3) or size.shape != (2,)
            or not np.isfinite(intrinsic).all() or not np.isfinite(size).all()
            or (size <= 0).any() or (np.diag(intrinsic)[:2] <= 0).any()):
        raise ValueError('Invalid estimated camera intrinsics/image size')
    np.testing.assert_allclose(intrinsic[2], [0, 0, 1], atol=1e-6)
    return c2w, intrinsic, size


class WorldTransform:
    """Apply one declared similarity to body, camera and scene together.

    This does not estimate an unknown metric scale or gravity. Callers must
    record how rotation/scale were obtained; pixel projections are preserved.
    Points use row storage, while camera matrices use column-vector c2w.
    """
    def __init__(self, rotation, scale, translation):
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.translation = np.asarray(translation, dtype=np.float64)
        self.scale = float(scale)
        if (self.rotation.shape != (3, 3) or self.translation.shape != (3,)
                or not np.isfinite(self.rotation).all()
                or not np.isfinite(self.translation).all()
                or not np.isfinite(self.scale) or self.scale <= 0):
            raise ValueError('Expected a finite proper rotation and positive world scale')
        np.testing.assert_allclose(self.rotation.T@self.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(self.rotation), 1., atol=1e-6)

    def points(self, points):
        return self.scale*(np.asarray(points)@self.rotation.T)+self.translation

    def camera_to_world(self, c2w):
        c2w = np.asarray(c2w)
        if c2w.shape[-2:] != (4, 4):
            raise ValueError('Expected [...,4,4] camera-to-world matrices')
        transformed = c2w.copy().astype(np.float64)
        transformed[..., :3, :3] = self.rotation@c2w[..., :3, :3]
        transformed[..., :3, 3] = self.points(c2w[..., :3, 3])
        return transformed


def projected_area(vertices, c2w, intrinsic, image_hw):
    local = (vertices-c2w[:3, 3])@c2w[:3, :3]
    present = np.isfinite(local).all(-1) & (local[:, 2] > 1e-5)
    if present.sum() < 3:
        return 0.
    pixel = local[present]@intrinsic.T
    pixel = pixel[:, :2]/pixel[:, 2:3]
    limit = image_hw[::-1]
    return float(np.maximum(np.clip(pixel.max(0), 0, limit)-np.clip(pixel.min(0), 0, limit), 0).prod())


def audit(archive, output, source_model):
    ensure_bounded()
    output.mkdir(parents=True, exist_ok=True)
    # Explicit native topology validation; no beta conversion or personal GT.
    import smplx
    model = smplx.SMPL(str(source_model), batch_size=1)
    regressor = model.J_regressor[:22].detach().numpy()
    faces = model.faces
    report = dict(state='auditing', archive=str(archive), archive_sha256=sha256(archive),
        started_at_utc=now(), ground_truth_used=False, training_labels_created=False,
        joint_definition='SMPL mesh regressor first 22 rows; not native posed SMPL joints',
        regressor_model=str(source_model), regressor_model_sha256=sha256(source_model),
        reader_sha256=sha256(Path(__file__)), camera_source='hsi',
        source_world_units_preserved=True, metric_scale_guaranteed=False,
        gravity_available=False, actual_joint_reliability_available=False, sequences=[])
    atomic_json(output/'STATUS.json', report)
    with ZipFile(archive) as z:
        names = set(z.namelist())
        roots = sorted({name.split('/')[0] for name in names})
        for root in roots:
            top = json.loads(z.read(root+'/manifest.json'))
            records = top['frames']
            if len(records) != top['num_frames']:
                raise ValueError('Top-level sequence count differs from frame list')
            people_by_frame, metadata, cameras, intrinsics, sizes = [], [], [], [], []
            areas, counts = Counter(), Counter()
            for row in records:
                path = PurePosixPath(root)/row['directory']
                if '..' in path.parts or path.is_absolute():
                    raise ValueError('Invalid sequence member path')
                prefix = path.as_posix()+'/'
                meta = json.loads(z.read(prefix+'manifest.json'))
                body = arrays(z.read(prefix+'smpl.pkl'))
                cs = arrays(z.read(prefix+'cameras.pkl'))
                pose, intrinsic, hw = camera(cs)
                source = re.search(r'_frame_(\d+)$', meta['frame_id'])
                if source is None:
                    raise ValueError('Missing source clock identity')
                clouds = meta['files']['environment'].get('hsi')
                if clouds and prefix+clouds['file'] not in names:
                    raise ValueError('Referenced scene cloud missing from archive')
                candidates = [p for p in body['people'] if p['kind'] == 'hsi']
                for p in candidates:
                    np.testing.assert_array_equal(p['faces'], faces)
                    vertices = np.asarray(p['vertices_world'])
                    if vertices.shape != (6890, 3) or not np.isfinite(vertices).all():
                        raise ValueError('Invalid estimated mesh')
                    key = str(p['track_id'])
                    counts[key] += 1
                    areas[key] += projected_area(vertices, pose, intrinsic, hw)
                people_by_frame.append(candidates)
                metadata.append(dict(source_frame=int(source.group(1)), frame_id=meta['frame_id'],
                    scene_member=prefix+clouds['file'] if clouds else None,
                    scene_points=clouds['point_count'] if clouds else 0,
                    source_world_units=meta['world_units'], settings=meta['settings']))
                cameras.append(pose); intrinsics.append(intrinsic); sizes.append(hw)
            if not areas:
                raise ValueError('No estimated target track')
            selected = max(sorted(areas), key=areas.get)
            joints = np.full((len(records), 22, 3), np.nan, np.float32)
            valid = np.zeros(len(records), bool)
            for index, people in enumerate(people_by_frame):
                found = [p for p in people if str(p['track_id']) == selected]
                if len(found)>1:
                    raise ValueError('Ambiguous track in one frame')
                if found:
                    joints[index] = regressor@found[0]['vertices_world']
                    valid[index] = True
            ids = np.array([row['source_frame'] for row in metadata], np.int64)
            if not (np.diff(ids)>0).all():
                raise ValueError('Source frame IDs are duplicated or unordered')
            missing_clock = sorted(set(range(int(ids[0]), int(ids[-1])+1))-set(ids.tolist()))
            array_path = output/(root+'.npz')
            np.savez_compressed(array_path, source_frame_ids=ids, joints_world_unscaled=joints,
                body_available=valid, camera_c2w_unscaled=np.asarray(cameras),
                intrinsics=np.asarray(intrinsics), image_hw=np.asarray(sizes))
            record = dict(sequence=root, exported_frames=len(records), selected_track=selected,
                selection='largest summed in-image projected estimated-mesh bbox; no GT',
                per_track_frames=dict(counts), body_available_frames=int(valid.sum()),
                body_missing_source_frames=ids[~valid].tolist(), source_clock_missing_frames=missing_clock,
                per_joint_visibility_known=False, pose_and_betas_required_for_joint_input=False,
                mesh_regressed_joints_are_estimates=True, metric_ready=False,
                npz=str(array_path), npz_sha256=sha256(array_path))
            atomic_json(output/(root+'.json'), dict(**record, frames=metadata))
            report['sequences'].append(record)
            atomic_json(output/'STATUS.json', report)
            print(json.dumps(record), flush=True)
    report.update(state='complete', completed_at_utc=now(),
        total_exported_frames=sum(s['exported_frames'] for s in report['sequences']),
        inputs_ready_as_unscaled_world_tracks=True, inference_not_run=True,
        remaining_coordinate_contract='One common scale/up/clock transform for body, camera and scene; never use GT alignment')
    atomic_json(output/'RESULT.json', report)
    atomic_json(output/'STATUS.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smpl-model', type=Path, required=True,
                   help='Local native SMPL model with matching mesh topology; never a personal GT fit')
    args = p.parse_args()
    audit(args.archive, args.output, args.smpl_model)


if __name__ == '__main__':
    main()
