# Experiment storage and reproducibility

The user asks that storage cleanup be part of experiment work, without repeated reminders.

- Inventory disk usage before and after substantial experiments. Checkpoint files consume disk; distinguish this from training RAM/VRAM.
- New short trials overwrite one atomic `last.pt`; do not automatically create `best.pt` or numbered checkpoint copies when evaluation uses the final step.
- After a trial finishes and evaluation is recorded, keep its model, configuration, metrics, identities, seeds, and audit evidence. Strip optimizer tensors from rejected finished trials when no resume is planned; mark the file evaluation-only, and preserve small RNG/sampler states needed by audits.
- Keep the official 55k model and the currently used warm-start/control checkpoint resumable. Preserve selected useful candidates. Verify retained model tensors before removing unused checkpoints, and log the cleanup.
- Delete only identified redundant/rebuildable artifacts. Native source archives, dataset manifests, contact labels, body models, and scene caches are active training dependencies even if they live in another repository. Audit dependencies before removing preprocessing data.
- Large weights and generated arrays stay local; back up code and compact experimental records to the authorized user GitHub repository.
