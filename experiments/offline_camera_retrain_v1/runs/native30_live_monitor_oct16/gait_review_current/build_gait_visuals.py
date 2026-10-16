"""CPU-only replay and foot traces from one frozen real validation archive."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.data import PARENTS

FOLDER = Path(__file__).resolve().parent


def make_plots(case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = np.arange(len(case["joints"]["truth"])) / 30
    truth = np.asarray(case["joints"]["truth"])
    for regime in ("clean", "noisy"):
        prediction = np.asarray(case["joints"][regime])
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for foot, color, side in ((10, "#e47d21", "Left"), (11, "#1c9a69", "Right")):
            for joints, label, linestyle, alpha in ((truth, "GT", "-", .9), (prediction, "Pred", "--", .8)):
                p = joints[:, foot]
                axes[0, 0].plot(p[:, 0], p[:, 2], linestyle, color=color, alpha=alpha, label=f"{side} {label}")
                axes[0, 1].plot(time, p[:, 1], linestyle, color=color, alpha=alpha, label=f"{side} {label}")
                speed = np.linalg.norm(np.diff(p, axis=0), axis=-1) * 30
                axes[0, 2].plot(time[1:], speed, linestyle, color=color, alpha=alpha, label=f"{side} {label}")
                local = p - joints[:, 0]
                axes[1, 0].plot(time, local[:, 2], linestyle, color=color, alpha=alpha, label=f"{side} {label}")
            # Plot predicted displacement only during contiguous GT low-speed intervals.
            low = np.asarray(case["gt_low_speed"])[1:, foot - 10]
            speed = np.linalg.norm(np.diff(prediction[:, foot], axis=0), axis=-1) * 30
            axes[1, 1].plot(time[1:], np.where(low, speed, np.nan), color=color, label=f"{side} pred speed in GT low-speed")
        for joints, color, label in ((truth, "#2574cf", "GT pelvis"), (prediction, "#c32f8e", "Pred pelvis")):
            root = joints[:, 0]
            displacement = np.linalg.norm(root[:, (0, 2)] - root[0, (0, 2)], axis=-1)
            axes[1, 2].plot(time, displacement, color=color, label=label)
        axes[0, 0].set(xlabel="X (m)", ylabel="Z (m)", title="Absolute window foot trajectories")
        axes[0, 0].axis("equal")
        axes[0, 1].set(xlabel="Time (s)", ylabel="Y up (m)", title="Absolute FK foot-joint height")
        axes[0, 2].set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Absolute foot-joint 3D speed")
        axes[0, 2].axhline(.2, color="gray", lw=1, ls=":")
        axes[1, 0].set(xlabel="Time (s)", ylabel="Foot Z - pelvis Z (m)", title="Pelvis-relative leg motion (not foot sliding)")
        axes[1, 1].set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Predicted foot speed where GT speed < 0.2 m/s")
        axes[1, 1].axhline(.2, color="gray", lw=1, ls=":")
        axes[1, 2].set(xlabel="Time (s)", ylabel="Horizontal displacement (m)", title="Pelvis displacement from window start")
        for ax in axes.flat:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(f"Actual validation step {case['step']} / {case['dataset']} / {case['panel']} / {regime}\n"
            f"{case['sequence_id']} | Native FK joints, no alignment or floor snap", fontsize=12)
        fig.tight_layout(rect=(0, .025, 1, .94))
        path = FOLDER / f"FOOT_TRACES_case{case['case']}_{regime}.png"
        fig.savefig(path, dpi=115); plt.close(fig)


def main():
    torch.set_num_threads(1)
    provenance = json.loads((FOLDER / "PROVENANCE.json").read_text())
    record = json.loads((FOLDER / "FROZEN_RECORD.json").read_text())
    snapshot = FOLDER / "FROZEN_VALIDATION.npz"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != provenance["source_archive_sha256"]:
        raise RuntimeError("Frozen archive digest changed")
    with np.load(snapshot, allow_pickle=False) as a:
        if int(a["step"]) != provenance["step"] or record["step"] != provenance["step"]:
            raise RuntimeError("Frozen step attribution changed")
        identities = json.loads(str(a["identities_json"]))
        cases = []
        for row in record["cases"]:
            if row["regime"] != "clean" or row["mode"] != "ego" or row.get("scene_variant", "scene_on") != "scene_on":
                continue
            i = row["case"]; identity = identities[i]
            assert identity["sequence_id"] == row["sequence_id"] and identity["start"] == row["start"]
            rest = torch.from_numpy(a[f"{i}_rest"])[None]
            joints = {}
            maximum_stored_FK_error = 0.
            for name, key in (("truth", f"{i}_truth"), ("clean", f"{i}_clean_prediction"), ("noisy", f"{i}_noisy_prediction")):
                motion = a[key]
                with torch.no_grad():
                    j = forward_kinematics(torch.from_numpy(motion)[None], rest)[0].numpy()
                if not np.isfinite(j).all():
                    raise RuntimeError("Non-finite actual FK output")
                if name == "truth":
                    maximum_stored_FK_error = float(np.abs(j - motion[:, 135:].reshape(len(motion), 22, 3) * 2).max())
                    if maximum_stored_FK_error > 1e-4:
                        raise RuntimeError("Stored GT and native FK disagree")
                joints[name] = np.round(j, 6).tolist()
            truth = np.asarray(joints["truth"])
            speed = np.linalg.norm(np.diff(truth[:, (10, 11)], axis=0), axis=-1) * 30
            low = np.r_[speed[:1], speed] < .2
            case = dict(case=i, step=provenance["step"], dataset=row["dataset"], panel=row["panel"],
                sequence_id=identity["sequence_id"], source_start_30fps=identity["source_start_30fps"],
                start=identity["start"], origin_world_m=identity["origin"], window_anchor=identity["anchor"],
                joints=joints, gt_low_speed=low.tolist(), GT_stored_FK_max_error_m=maximum_stored_FK_error,
                metrics={regime: next(r["metrics"] for r in record["cases"] if r["case"] == i and r["regime"] == regime
                    and r.get("scene_variant", "scene_on") == "scene_on") for regime in ("clean", "noisy")})
            cases.append(case)
            make_plots(case)
            print(json.dumps(dict(event="rendered", step=provenance["step"], case=i, dataset=row["dataset"], panel=row["panel"])), flush=True)
    payload = dict(step=provenance["step"], fps=30, frames=192, parents=list(PARENTS), cases=cases,
        source_archive_sha256=provenance["source_archive_sha256"], pose_alignment=False, floor_snap=False)
    template = (FOLDER / "replay_template.html").read_text()
    (FOLDER / "REPLAY.html").write_text(template.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
    summary = dict(step=provenance["step"], source_archive_sha256=provenance["source_archive_sha256"],
        cases=[{key: value for key, value in case.items() if key not in ("joints", "gt_low_speed")} for case in cases],
        replay="REPLAY.html", figures=[str(path.name) for path in FOLDER.glob("FOOT_TRACES*.png")],
        geometry="Exact native differentiable FK joints from frozen output poses and corresponding native rest shape; feet joints10/11, not full skin mesh soles",
        coordinates="Absolute shared window coordinates by default: rigid world-to-window transform only. Optional pelvis-relative display subtracts each body's own pelvis; unsuitable for judging absolute foot sliding.",
        low_speed="GT foot speed<0.2m/s, same definition as existing validation; low speed alone is not ground-truth contact and seated windows are not locomotion evidence",
        optimizer_steps=0, model_forward_calls=0, cuda_used=False, training_unchanged=True,
        build_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (FOLDER / "VISUALIZATION_MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
