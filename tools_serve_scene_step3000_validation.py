"""Serve the prepared step-3000 cross-dataset validation cases in MotionViewer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer, atomic_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/scene_joint128_conditioning_corrected/validation_visualizations_step003000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--manifest", type=Path, default=OUT / "CASES.json")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    viewer = MotionViewer(cases, args.port)
    # Show the complete static GRAIL reference by default, while keeping the
    # causal observed map visible.  The blue pelvis trajectory is hidden at
    # startup so it cannot be mistaken for a scene surface; it remains
    # available through the existing path checkbox.
    for panel in viewer.panels.values():
        panel["scan"].value = True
        panel["static"].value = True
        panel["paths"].value = False
    viewer.update()
    atomic_json(output_dir / "SERVER.json", dict(
        pid=os.getpid(), url=f"http://localhost:{args.port}/", bind="127.0.0.1",
        cases=len(cases), viewer="experiments.offline_camera_retrain_v1.ops.motion_viewer.MotionViewer",
        manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        checkpoint_step=manifest.get("checkpoint_step"), test_used=manifest.get("test_used", False)))
    print(f"VISER_URL=http://localhost:{args.port}/", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
