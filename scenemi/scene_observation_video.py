"""Project saved first-hit scene history through its original EGO camera."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def export(manifest, case_index, output):
    import cv2
    from scenemi.training import ensure_bounded
    from scenemi.data.amass_prepare import atomic_json, sha256
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    ensure_bounded()
    case = json.loads(manifest.read_text())['cases'][case_index-1]
    with np.load(case['models'][0]['case']['cache'], allow_pickle=False) as saved:
        identity = json.loads(str(saved['metadata_json']))['identity']
    folder = Path(identity['path'])
    meta = dict(json.loads((folder/'metadata.json').read_text()), path=str(folder))
    read = lambda name: np.asarray(readonly_array(meta, name))
    points, owners = read('visible_frame_points_scenemi_yup'), read('visible_frame_owner')
    positions, rotations, intrinsics = [read('camera_'+name) for name in (
        'position_scenemi_yup', 'rotation_scenemi_yup', 'intrinsics_128x96')]
    valid_frames = read('observation_valid').astype(bool)
    start = identity['start']; stop = start+128
    origin, anchor = np.asarray(identity['origin']), np.asarray(identity['anchor'])
    transform = lambda x: ((np.asarray(x)-origin)@anchor.T).astype(np.float32)
    camera = transform(positions)
    world_points = transform(points)
    camera_rotation = anchor@rotations
    # Only the fixed overview camera uses motion bounds. EGO always uses the saved camera.
    center = camera[start:stop].mean(0); center[1] = .5
    eye = center+np.array([3.2, 3., 3.6])
    forward = (center-eye); forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    overview_rotation = np.stack((right, up, forward), axis=1)
    history, seen, counts = [], set(), []
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1640, 800
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
        '-video_size', f'{width}x{height}', '-framerate', '30', '-i', '-', '-an',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def text(image, value, xy, scale=.65, color=(225, 231, 236)):
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def project(p, position, rotation, k, scale=1):
        local = (p-position)@rotation
        z = local[:, 2]
        keep = (z > .05) & (z < (meta['virtual_camera']['far_m'] if scale == 6 else 50))
        local, z = local[keep], z[keep]
        uv = np.stack((k[0]*local[:, 0]/z+k[2], k[3]-k[1]*local[:, 1]/z), -1)*scale
        inside = (uv[:, 0]>=0)&(uv[:, 0]<768)&(uv[:, 1]>=0)&(uv[:, 1]<576)
        return uv[inside].astype(int), z[inside], np.flatnonzero(keep)[inside]

    def cloud(image, p, position, rotation, k, scale=1, current=False):
        if not len(p):
            return
        uv, depth, ids = project(p, position, rotation, k, scale)
        for i in np.argsort(depth)[::-1]:
            color = (20, 210, 255) if current else ((235, 190, 35) if p[ids[i], 1]>.12 else (125, 105, 75))
            cv2.circle(image, tuple(uv[i]), 2 if current else 1, color, -1)

    try:
        for t in range(stop):
            current = world_points[t][(owners[t] == 0)&np.isfinite(world_points[t]).all(-1)] if valid_frames[t] else np.empty((0,3))
            for p in current:
                key = tuple(np.floor(p/.025).astype(np.int64))
                if key not in seen:
                    seen.add(key); history.append(p)
            if t < start:
                continue
            accumulated = np.asarray(history, np.float32).reshape(-1,3)
            canvas = np.full((height, width, 3), (26, 22, 19), np.uint8)
            left = canvas[100:676, 24:792]; right_image = canvas[100:676, 848:1616]
            cloud(left, accumulated, camera[t], camera_rotation[t], intrinsics[t], scale=6)
            cloud(left, current, camera[t], camera_rotation[t], intrinsics[t], scale=6, current=True)
            overview_k = [650., 650., 384., 288.]
            cloud(right_image, accumulated, eye, overview_rotation, overview_k)
            cloud(right_image, current, eye, overview_rotation, overview_k, current=True)
            uv, _, ids = project(camera[start:t+1], eye, overview_rotation, overview_k)
            if len(uv)>1:
                cv2.polylines(right_image, [uv.astype(np.int32)], False, (65, 105, 250), 2, cv2.LINE_AA)
            if len(uv):
                cv2.circle(right_image, tuple(uv[-1]), 6, (65, 105, 250), -1)
            k = intrinsics[t]
            corners = np.array([[(x-k[2])/k[0], (k[3]-y)/k[1], 1.] for x,y in [(0,0),(128,0),(128,96),(0,96)]])*.45
            corners = corners@camera_rotation[t].T+camera[t]
            for a,b in [(camera[t], p) for p in corners]+[(corners[i],corners[(i+1)%4]) for i in range(4)]:
                line,_,_ = project(np.stack((a,b)), eye, overview_rotation, overview_k)
                if len(line)==2:
                    cv2.line(right_image,tuple(line[0]),tuple(line[1]),(45,145,250),1,cv2.LINE_AA)
            text(canvas, '01 | Original EGO camera + progressively observed scene', (24,35), .9)
            text(canvas, 'EGO projection: original pose, intrinsics and field of view', (24,78), .6)
            text(canvas, 'Overview: accumulated points + camera path + current frustum', (848,78), .6)
            text(canvas, f'Frame {t-start:03d}/127 | source frame {t} | {(t-start)/30:.2f}s | history: {len(history):,} points', (24,713), .75)
            text(canvas, 'Yellow: current first-hit samples | Cyan / muted: past observed surfaces | Orange: camera', (24,748), .65)
            text(canvas, 'Causal display: observations 0..current frame only. GRAIL exported virtual head camera; no recorded RGB.', (24,778), .6)
            process.stdin.write(canvas.tobytes())
            counts.append(len(history))
            if t-start in (0,32,64,96,127):
                cv2.imwrite(str(output.with_name(output.stem+f'_frame{t-start:03d}.png')),canvas)
        process.stdin.close()
        error = process.stderr.read().decode(); returncode = process.wait()
        if returncode:
            raise RuntimeError(error)
    except BaseException:
        process.kill(); process.wait(); raise
    assert all(b>=a for a,b in zip(counts,counts[1:]))
    atomic_json(output.with_suffix('.json'),dict(video=str(output.resolve()),sha256=sha256(output),
        source_metadata_sha256=sha256(folder/'metadata.json'), source_arrays_sha256={k:v for k,v in meta['output_sha256'].items()
            if k in ('visible_frame_points_scenemi_yup.npy','visible_frame_owner.npy','camera_position_scenemi_yup.npy',
                'camera_rotation_scenemi_yup.npy','camera_intrinsics_128x96.npy','observation_valid.npy')},
        identity=identity,frames=128,fps=30,point_counts=counts,voxel_m=.025,
        no_future_observations=True,source_mesh_used=False,original_camera=True,
        camera_kind='GRAIL exported virtual head camera, not recorded RGB',
        note='This causal display differs from the trained 17k offline whole-recording map.'))
    print('Exported',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.manifest,args.case,args.output)


if __name__=='__main__':
    main()
