"""Manifest-driven paired scene-on/off previews of retained EGO checkpoints."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def prepare(checkpoint, selection, output, device):
    import torch
    from scenemi.checkpoint import load_trained_model
    from scenemi.config import ROOT, load_paths
    from scenemi.data.amass_prepare import atomic_json, sha256
    from scenemi.data.scene_ego import SceneEgoData, collate, payload
    from scenemi.data.grail_chair import precheck
    from scenemi.data.grail_filter import load_policy
    from scenemi.geometry.native_shape import GENDERS
    from scenemi.replay.visualization import mesh, display_geometry
    from scenemi.training import deterministic, ensure_bounded
    from experiments.offline_camera_retrain_v1.native30_model import sample_native30

    ensure_bounded()
    if output.exists():
        raise ValueError('Choose a new output directory')
    spec = json.loads(selection.read_text())
    deterministic(spec['seed'], device)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(6 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, c, step = load_trained_model(checkpoint)
    if c['version'] != 'ego-scene-adaptation-v1' or step != spec['checkpoint_step']:
        raise ValueError('Expected the declared trained scene checkpoint')
    data = SceneEgoData(c, load_paths(ROOT / 'configs/paths.local.yaml'), 'validation')
    model = model.to(device)
    digest = sha256(checkpoint)
    output.mkdir(parents=True)
    atomic_json(output / 'SELECTION.json', dict(spec, checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=digest, selection_sha256=sha256(selection), test_used=False,
        source_sha256=sha256(__file__), device=device, paired_noise=True,
        target_shape_used_for_prediction=False))
    prepared, summary, meshes = [], [], {}
    policy = load_policy(ROOT / 'configs/grail_ego_support_v2.json')
    with torch.inference_mode():
        for index, chosen in enumerate(spec['cases']):
            case = chosen['case']
            sample, identity = data.sample_case(case)
            if identity != chosen['identity']:
                raise ValueError('Saved validation identity differs')
            meta = data.native.selection_bundle.metadata(data.native.records[case['member']], 'validation')
            visibility = dict(manual_selection_note=chosen.get('note', ''))
            if identity['dataset'] == 'grail':
                visibility, _, _ = precheck(meta, policy)
                pixels = np.array([r['object_above_ground'] for r in meta['primary_visibility_by_frame']])
                onset = visibility['first_stable_sit_frame']
                visibility.update(object_pixels_before_sit=int(pixels[:onset].sum()),
                    frames_above_123_pixels_before_sit=int((pixels[:onset] >= 123).sum()))
            batch = collate([sample], device)
            public = payload(batch)
            # The exact same payload and sampler seed are reused for both arms.
            raw, _ = data.native.sample(np.random.default_rng(777 + case['case_id']),
                member=case['member'], start=case['start'], yaw=0.)
            tv, faces, tj, te = mesh(sample['motion'], sample['target_rest'], meta['native_body'], meshes)
            geometry = display_geometry(data.native.selection_bundle, meta, dict(start=case['start']), tj, 'ego')
            geometry['visible_static'] = raw['local_history_points'][raw['local_history_valid']]
            cp = sample['camera'][:, :3] * 2
            six = sample['camera'][:, 3:]
            rotation = np.stack((six[:, :3], six[:, 3:], np.cross(six[:, :3], six[:, 3:])), -1)
            ends = cp[:, None] + rotation.swapaxes(-1, -2) * .3
            models, arms = [], {}
            for enabled, key, label, color in ((True, 'scene_on', '17k · 有场景', [185, 40, 160]),
                                                (False, 'scene_off', '17k · 无场景', [235, 140, 35])):
                prediction = sample_native30(model, public, steps=c['validation']['ddim_steps'],
                    seed=spec['seed'], use_scene=enabled)
                _, details = model(prediction, torch.zeros(1, dtype=torch.long, device=device), public,
                    use_scene=enabled, return_details=True)
                rest = details['rest'][0].cpu().numpy()
                info = dict(meta['native_body'], gender=GENDERS[int(details['predicted_template'][0])],
                    betas=details['predicted_native_betas'][0].cpu().numpy().tolist())
                motion = prediction[0].cpu().numpy()
                pv, pf, pj, pe = mesh(motion, rest, info, meshes)
                if not np.array_equal(faces, pf):
                    raise ValueError('Body topologies disagree')
                delta = pj[:, 0] - tj[:, 0]
                metrics = dict(world_mpjpe_mm=float(np.linalg.norm(pj-tj, axis=-1).mean()*1000),
                    root_mean_error_mm=float(np.linalg.norm(delta, axis=-1).mean()*1000),
                    root_start_error_mm=float(np.linalg.norm(delta[0])*1000),
                    root_end_error_mm=float(np.linalg.norm(delta[-1])*1000),
                    root_error_vector_change_mm=float(np.linalg.norm(delta[-1]-delta[0])*1000),
                    mesh_fk_max_error_m=max(pe, te),
                    final_root_height_m=float(pj[-16:, 0, 1].mean()),
                    final_knee_height_m=float(pj[-16:, [4, 5], 1].mean()))
                artifact = output / f'case_{index+1:02d}_{key}.npz'
                np.savez_compressed(artifact, truth_vertices=tv, truth_joints=tj, faces=faces,
                    preview_vertices=pv, preview_joints=pj, preview_reference_vertices=pv,
                    preview_reference_joints=pj, preview_observations=sample['trajectory'],
                    preview_observation_meta=sample['observation_meta'], prediction_motion=motion,
                    prediction_rest=rest, truth_motion=sample['motion'],
                    source_occluded=np.zeros((128, 22), bool),
                    center=tj[:, 0].mean(0)+np.array([0, .4, 0]), radius=np.float32(2.6),
                    camera_positions=cp, camera_axis_segments=np.stack((np.broadcast_to(cp[:, None], ends.shape), ends), 2),
                    camera_frustum_segments=np.empty((128, 0, 2, 3), np.float32),
                    camera_path=np.stack((cp[:-1], cp[1:]), 1),
                    metadata_json=np.array(json.dumps(dict(identity=identity, metrics=metrics,
                        checkpoint_step=step, checkpoint_sha256=digest, scene_fed_to_model=enabled,
                        target_shape_used_for_prediction=False, scene_errors=[]))), **geometry)
                models.append(dict(key=key, label=label, color=color, mode='ego',
                    case=dict(cache=str(artifact.resolve()), cache_sha256=sha256(artifact))))
                arms[key] = metrics
                print(f'{index+1}/{len(spec["cases"])} {key}: MPJPE {metrics["world_mpjpe_mm"]:.2f} mm', flush=True)
            record = dict(sequence_id=identity['sequence_id'], start=case['start'],
                frames=128, visibility=visibility, arms=arms)
            summary.append(record)
            prepared.append(dict(label=chosen['label'], display_label=chosen['label'],
                regimes=['preview'], models=models, scene_fed_to_model=True,
                selection_reason=f'{identity["sequence_id"]} · 原帧 {case["start"]}–{case["start"]+127}。'
                    '同一 17k 权重、同一相机输入、同一噪声；仅切换场景分支。蓝色 GT；紫色有场景；橙色无场景。'))
    atomic_json(output/'RESULT.json', dict(checkpoint_step=step, checkpoint_sha256=digest,
        device=device, seed=spec['seed'], ddim_steps=c['validation']['ddim_steps'], cases=summary,
        scene_policy='offline whole-recording observed first-hit map; noncausal; no full mesh input',
        inference_only=True, test_used=False, selection=spec['selection']))
    atomic_json(output/'PREPARED_CASES.json', dict(protocol='scene-paired-case-preview-v1', cases=prepared))


def serve(output, port, manifest=None):
    from scenemi.data.amass_prepare import atomic_json, sha256
    from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer
    manifest = manifest or output/'PREPARED_CASES.json'
    value = json.loads(manifest.read_text())
    for case in value['cases']:
        for entry in case['models']:
            if sha256(entry['case']['cache']) != entry['case']['cache_sha256']:
                raise ValueError('Preview artifact changed')

    class PairedViewer(MotionViewer):
        def load_case(self):
            super().load_case()
            self.panels['ego']['scan'].value = True
            self.panels['ego']['static'].value = False
            self.panels['ego']['current'].value = True
            self.panels['ego']['camera_path'].value = True
            for mode, panel in self.panels.items():
                for key in ('reference', 'uncertainty', 'reference_source', 'occluded'):
                    panel[key].visible = False
                panel['reference'].value = panel['uncertainty'].value = False
                for button in self.preset_buttons[mode].values():
                    button.visible = False
            self.regime.visible = False
            self.update()

        def update(self):
            super().update()
            if self.loading or 'ego' not in self.runtime:
                return
            self.panels['ego']['info'].content = (
                '**同一 17k checkpoint**，DDIM20、相同 seed、仅相机轨迹；体型由模型预测。\n\n'
                '紫色：开启场景。橙色：关闭场景。蓝色：GT。勾选图层可分别看动作。\n\n'
                '场景输入是整段录制中实际观测点的并集（离线、非因果）。完整椅子网格仅作显示参考。\n\n' +
                '\n\n'.join(f'{item["entry"]["label"]}：MPJPE {item["meta"]["metrics"]["world_mpjpe_mm"]:.1f} mm'
                    for item in self.runtime['ego']['models']))

    viewer = PairedViewer(value['cases'], port)
    atomic_json(output/'SERVER.json', dict(pid=os.getpid(), url=f'http://localhost:{port}/',
        manifest=str(manifest.resolve())))
    print(f'READY http://localhost:{port}/', flush=True)
    viewer.run()


def refresh_layers(manifest):
    """Add original current observations and real camera glyphs without resampling motion."""
    from scenemi.data.amass_prepare import atomic_json, sha256
    from scenemi.training import ensure_bounded
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import camera_layers, instant_scene_layers
    ensure_bounded()
    value = json.loads(manifest.read_text())
    hashes, records = {}, []
    for case in value['cases']:
        layers = None
        for entry in case['models']:
            file = Path(entry['case']['cache'])
            with np.load(file, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            metadata = json.loads(str(arrays['metadata_json']))
            identity = metadata['identity']
            if layers is None:
                source = dict(json.loads((Path(identity['path'])/'metadata.json').read_text()),
                    path=identity['path'])
                source.update(identity)
                current, current_note = instant_scene_layers(source, len(arrays['truth_joints']))
                camera, camera_note = camera_layers(source, len(arrays['truth_joints']))
                layers = dict(current, **camera)
                records.append(dict(sequence_id=identity['sequence_id'], start=identity['start'],
                    current_scene=current_note, camera=camera_note))
            arrays.update(layers)
            metadata['display_geometry'] = dict(current_scene=current_note, camera=camera_note)
            arrays['metadata_json'] = np.array(json.dumps(metadata))
            before = {k: sha256_bytes(arrays[k]) for k in ('prediction_motion', 'preview_joints', 'truth_joints')}
            temporary = file.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            with np.load(temporary, allow_pickle=False) as z:
                assert all(sha256_bytes(z[k]) == digest for k, digest in before.items())
            temporary.replace(file)
            hashes[str(file)] = sha256(file)
    # Update all prepared manifests referring to these same exported arrays.
    for parent in {Path(p).parent for p in hashes} | {manifest.parent}:
        for file in parent.glob('*CASES.json'):
            saved = json.loads(file.read_text())
            changed = False
            for case in saved.get('cases', []):
                for entry in case.get('models', []):
                    record = entry.get('case', {})
                    if record.get('cache') in hashes:
                        record['cache_sha256'] = hashes[record['cache']]; changed = True
            if changed:
                atomic_json(file, saved)
    atomic_json(manifest.parent/'CURRENT_LAYERS_VALIDATION.json', dict(passed=True,
        prediction_arrays_unchanged=True, source='saved original first-hit samples and observer transforms', cases=records))
    print('Restored current scene and camera layers:', len(records), 'cases', flush=True)


def sha256_bytes(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def render(output, stills_only=False):
    """Export a fixed-camera three-column movie and final-pose still per case."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scenemi.training import ensure_bounded
    ensure_bounded()
    value = json.loads((output/'PREPARED_CASES.json').read_text())
    for index, case in enumerate(value['cases']):
        arrays = []
        for entry in case['models']:
            with np.load(entry['case']['cache'], allow_pickle=False) as z:
                arrays.append({k: z[k] for k in ('truth_vertices', 'preview_vertices', 'truth_joints',
                    'preview_joints', 'faces', 'scene_vertices', 'scene_faces', 'metadata_json')})
        first = arrays[0]
        gt = first['truth_vertices']
        low = gt.min(axis=(0, 1)); high = gt.max(axis=(0, 1))
        center = (low + high) / 2
        span = max(2.2, float((high-low)[[0, 2]].max()) + .6)
        bounds = [(center[i]-span/2, center[i]+span/2) for i in (0, 2)] + [(0., max(2.05, float(high[1])+.1))]
        sv, sf = first['scene_vertices'], first['scene_faces']
        centroid = sv[sf].mean(1)
        keep = (np.abs(centroid[:, 0]-center[0]) < span/2+.2) & (np.abs(centroid[:, 2]-center[2]) < span/2+.2)
        sf = sf[keep]
        # Rendering samples only the display mesh; inference arrays stay exact.
        if len(sf) > 12000:
            sf = sf[np.linspace(0, len(sf)-1, 12000, dtype=int)]
        fig = plt.figure(figsize=(13.5, 5.2), facecolor='white')
        names = ['Ground truth', '17k + observed scene', '17k / scene disabled']
        dataset = json.loads(str(first['metadata_json']))['identity']['dataset'].upper()
        colors = ['#397bce', '#b72b9f', '#e79623']
        motions = [gt, arrays[0]['preview_vertices'], arrays[1]['preview_vertices']]
        artists = []
        for panel, (name, color, motion) in enumerate(zip(names, colors, motions)):
            ax = fig.add_subplot(1, 3, panel+1, projection='3d', computed_zorder=False)
            ax.add_collection3d(Poly3DCollection(sv[sf][..., [0, 2, 1]], facecolor='#b9c0bd',
                edgecolor='none', alpha=.4, zorder=1))
            artist = Poly3DCollection(motion[0][first['faces']][..., [0, 2, 1]],
                facecolor=color, edgecolor='none', alpha=1, zorder=2)
            ax.add_collection3d(artist); artists.append(artist)
            ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
            ax.set_box_aspect((span, span, bounds[2][1]))
            ax.view_init(elev=15, azim=-65)
            ax.dist = 7
            ax.set_axis_off()
            extra = '' if panel == 0 else '\nMPJPE %.1f mm' % json.loads(str(arrays[panel-1]['metadata_json']))['metrics']['world_mpjpe_mm']
            ax.set_title(name+extra, color=color, fontsize=13, pad=0)
        title = fig.suptitle(f'{dataset} validation case {index+1} | same camera, weights and noise (seed 777)', fontsize=13)
        fig.text(.5, .035, 'Scene: whole-recording observed points (offline, noncausal). Full chair mesh shown only as reference.',
            ha='center', fontsize=10)
        fig.subplots_adjust(left=0, right=1, bottom=.07, top=.84, wspace=0)
        for t, suffix in ((64, 'mid'), (127, 'final')):
            for artist, motion in zip(artists, motions):
                artist.set_verts(motion[t][first['faces']][..., [0, 2, 1]])
            title.set_text(f'{dataset} validation case {index+1} | frame {t} | same camera, weights and seed 777')
            fig.savefig(output/f'case_{index+1:02d}_{suffix}.png', dpi=140)
        if stills_only:
            plt.close(fig)
            continue
        movie = output / f'case_{index+1:02d}_comparison.mp4'
        writer = FFMpegWriter(fps=15, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'])
        with writer.saving(fig, str(movie), dpi=100):
            for t in range(0, 128, 2):
                for artist, motion in zip(artists, motions):
                    artist.set_verts(motion[t][first['faces']][..., [0, 2, 1]])
                title.set_text(f'{dataset} validation case {index+1} | frame {t:03d}/127 | {t/30:.2f}s | seed 777')
                writer.grab_frame()
        for artist, motion in zip(artists, motions):
            artist.set_verts(motion[-1][first['faces']][..., [0, 2, 1]])
        title.set_text(f'{dataset} validation case {index+1} | final pose | same camera, weights and seed 777')
        fig.savefig(output/f'case_{index+1:02d}_final.png', dpi=140)
        plt.close(fig)
        print(f'Rendered {movie}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'serve', 'render', 'refresh-layers'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--port', type=int, default=8096)
    parser.add_argument('--stills-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.checkpoint is None or args.selection is None:
            parser.error('prepare requires --checkpoint and --selection')
        prepare(args.checkpoint, args.selection, args.output, args.device)
    elif args.action == 'serve':
        serve(args.output, args.port, args.manifest)
    elif args.action == 'refresh-layers':
        refresh_layers(args.manifest or args.output/'PREPARED_CASES.json')
    else:
        render(args.output, args.stills_only)


if __name__ == '__main__':
    main()
