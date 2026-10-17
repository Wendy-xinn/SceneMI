# Experiment storage and reproducibility

The user asks that storage cleanup be part of experiment work, without repeated reminders.

- Inventory disk usage before and after substantial experiments. Checkpoint files consume disk; distinguish this from training RAM/VRAM.
- New short trials overwrite one atomic `last.pt`; do not automatically create `best.pt` or numbered checkpoint copies when evaluation uses the final step.
- After a trial finishes and evaluation is recorded, keep its configuration, per-case metrics, identities, seeds, source hashes, and audit evidence. The user's October 17 instruction is records-only retention for rejected trials with no planned reuse: remove their weights, prediction arrays, and rebuildable visualization/evaluation caches after recording exact file identities and checking dependencies. Mark historical audits as pre-cleanup evidence, and retire viewer menus that depended on removed arrays. Do not retain a rejected model merely to strip its optimizer.
- Keep the official 55k model and the currently used warm-start/control checkpoint resumable. Preserve selected useful candidates and arrays required by active comparisons, metric checks, or tests. Verify retained models and arrays before removing unused artifacts, and log the cleanup.
- Delete only identified redundant/rebuildable artifacts. Native source archives, dataset manifests, contact labels, body models, and scene caches are active training dependencies even if they live in another repository. Audit dependencies before removing preprocessing data.
- Large weights and generated arrays stay local; back up code and compact experimental records to the authorized user GitHub repository.
- Reuse `experiments/offline_camera_retrain_v1/ops` and its module CLI for monitoring, evaluation review, and visualization. Put case selections in data manifests; do not create a new dated Python script for each run or preview.
- Future preparation must seal an explicit dependency set, including dynamic entry points, rather than globbing every Python file in the experiment root. Existing runs retain their original immutable source seals until they and dependent evaluations finish; do not remove a sealed file merely because it is outside the runtime import closure.
