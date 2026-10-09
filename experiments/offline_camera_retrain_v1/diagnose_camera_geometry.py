import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def face_normal(x):
    eye = (x[9929] + x[9448]) / 2
    eye_line = x[9448] - x[9929]
    nose_line = x[9120] - eye
    f = np.cross(nose_line, eye_line)
    return f / max(np.linalg.norm(f), 1e-8), eye


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--frame', type=int, default=-1)
    args = p.parse_args()
    v = np.load(args.clip / 'body_vertices_scenemi_yup.npy', mmap_mode='r')
    cam = np.load(args.clip / 'camera_position_scenemi_yup.npy')
    rot = np.load(args.clip / 'camera_rotation_scenemi_yup.npy')
    t = len(v) - 1 if args.frame < 0 else args.frame
    f, eye = face_normal(v[t])
    p0, R = cam[t], rot[t]
    tx, ty = np.tan(np.deg2rad(66.56 / 2)), np.tan(np.deg2rad(40.49 / 2))
    local = .7 * np.array([[-tx, -ty, 1], [tx, -ty, 1], [tx, ty, 1], [-tx, ty, 1]])
    corners = local @ R.T + p0
    fig = plt.figure(figsize=(14, 6))
    for j, (a, b, title) in enumerate([(0, 2, 'X-Z projection'), (0, 1, 'X-Y projection'), (2, 1, 'Z-Y projection')]):
        ax = fig.add_subplot(1, 3, j + 1)
        ax.scatter(*np.array([v[t, 9929], v[t, 9448], eye]).T[[a, b]], c=['magenta','magenta','gold'], s=35, label='eyes')
        ax.scatter(p0[a], p0[b], c='black', s=35, label='camera')
        for q in corners:
            ax.plot([p0[a], q[a]], [p0[b], q[b]], color='orangered', lw=1.5)
        ax.plot(np.r_[corners[:, a], corners[0, a]], np.r_[corners[:, b], corners[0, b]], color='orangered', lw=1.5)
        ax.arrow(p0[a], p0[b], .35 * R[a, 2], .35 * R[b, 2], color='purple', width=.001, head_width=.03, length_includes_head=True)
        ax.arrow(eye[a], eye[b], .35 * f[a], .35 * f[b], color='green', width=.001, head_width=.03, length_includes_head=True)
        ax.set_aspect('equal'); ax.grid(True); ax.set_xlabel('xyz'[a]); ax.set_ylabel('xyz'[b]); ax.set_title(title)
        if j == 0: ax.legend(fontsize=8)
    fig.suptitle(f'{args.clip.name} frame {t} | black=camera, gold=eye midpoint, purple=camera forward, green=face normal')
    fig.tight_layout(); out=args.clip/'camera_geometry_diagnostic.png'; fig.savefig(out, dpi=160); print(out)


if __name__ == '__main__': main()
