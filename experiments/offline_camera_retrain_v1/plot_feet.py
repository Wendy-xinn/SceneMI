"""Plot generated foot trajectories from a standalone gallery."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=int, default=0)
    parser.add_argument('--variants', nargs='+', required=True)
    args = parser.parse_args()
    payload = json.loads(args.gallery.read_text().split('const data=', 1)[1].split(';\nconst $=', 1)[0])
    case = payload['cases'][args.case]
    gt = np.asarray(case['tracks']['GT'])
    direction = gt[-1, 0, [0, 2]] - gt[0, 0, [0, 2]]
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    time = np.arange(len(gt)) / 20
    fig, axes = plt.subplots(4, len(args.variants), figsize=(6 * len(args.variants), 11),
                             squeeze=False, constrained_layout=True)
    for col, variant in enumerate(args.variants):
        predicted = np.asarray(case['tracks'][variant])
        for data, style, alpha, label in [(gt, '--', .55, 'GT'), (predicted, '-', 1., 'prediction')]:
            for joint, color, name in [(0, 'black', 'pelvis'), (10, 'tab:blue', 'left'), (11, 'tab:orange', 'right')]:
                axes[0, col].plot(time, data[:, joint][:, [0, 2]] @ direction,
                                  style, color=color, alpha=alpha, label=f'{label} {name}')
            separation = (data[:, 10, [0, 2]] - data[:, 11, [0, 2]]) @ direction
            axes[1, col].plot(time, separation, style, alpha=alpha, label=label)
            velocity = np.linalg.norm(np.diff(data[:, (10, 11)], axis=0)[:, :, [0, 2]], axis=-1)
            axes[2, col].plot(time[1:], velocity.min(axis=-1) * 100, style, alpha=alpha, label=label)
            for joint, color, name in [(10, 'tab:blue', 'left'), (11, 'tab:orange', 'right')]:
                axes[3, col].plot(time, data[:, joint, 1], style, color=color, alpha=alpha, label=f'{label} {name}')
        axes[0, col].set_title(variant)
        axes[2, col].axhline(1, color='gray', linestyle=':')
        for ax, ylabel in zip(axes[:, col], ['Along-path position (m)', 'L-R separation (m)',
                                            'Slower foot speed (cm/frame)', 'Foot joint height (m)']):
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Time (s)')
            ax.grid(alpha=.2)
            ax.legend(ncol=2, fontsize=8)
    # Keep each row on a shared vertical scale for honest visual comparisons.
    for row in axes:
        lo = min(ax.get_ylim()[0] for ax in row)
        hi = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)
    fig.suptitle(f'{case["group"]}: {case["identity"]["sequence_id"]} / {case["identity"]["source_start_30fps"]}\n'
                 'Dashed = raw GT joints; height is not a scene collision measurement')
    fig.savefig(args.output, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
