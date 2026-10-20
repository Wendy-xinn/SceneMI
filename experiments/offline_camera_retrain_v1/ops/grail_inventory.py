"""Audit downloaded GRAIL reconstructions and seal an asset-grouped inventory.

The scan reads one reconstruction at a time on CPU. It never opens CUDA,
retargets robot motions, or changes downloaded source files.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

import numpy as np

from experiments.offline_camera_retrain_v1.ops import grail_source


PROTOCOL = "grail-full-source-inventory-v1"


def canonical_asset_family(source_object_path):
    """Discard reconstruction-run prefixes while retaining the asset namespace.

    Ascending/descending reconstructions of one generated mesh have different
    upstream results directories. Conversely, similarly named stairs in
    SynStairsV2 and SynStairsV3 are separate assets. The suffix following the
    last '/mesh/' preserves both facts; part folders and sequence indices do
    not participate in the family identity.
    """
    if not isinstance(source_object_path, str) or "/mesh/" not in source_object_path:
        raise ValueError("Source object path must contain an explicit /mesh/ namespace")
    suffix = source_object_path.rsplit("/mesh/", 1)[1]
    parts = PurePosixPath(suffix).parts
    if len(parts) < 3 or any(part in ("..", ".") for part in parts) or suffix.startswith("/"):
        raise ValueError("Asset family must contain a namespace, asset, and mesh filename")
    return "grail/" + "/".join(parts)


def _atomic(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def scan(source_root, output):
    from experiments.offline_camera_retrain_v1.ops.grail_data import family_split

    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source_root / "DOWNLOAD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_entries = {entry["path"]: entry for entry in manifest["files"]}
    entries = sorted((entry for entry in manifest["files"] if "/recon/" in entry["path"]),
                     key=lambda entry: entry["path"])
    if not entries:
        raise ValueError("The source manifest contains no reconstructions")
    temporary = output / "inventory.jsonl.tmp"
    started = time.time()
    category_stats, family_categories, family_splits = {}, defaultdict(set), {}
    total_frames = target_frames = recon_bytes = 0
    source_objects = set()
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, entry in enumerate(entries):
            path = source_root / entry["path"]
            category, sequence = path.parts[-3], path.stem
            record = grail_source.load_pickle(path)
            human, obj = record["human_data"], record["obj_data"]
            n = len(human["poses"])
            if human["poses"].shape != (n, 165) or human["trans"].shape != (n, 3):
                raise ValueError("Unexpected body schema: " + str(path))
            if not all(np.isfinite(human[key]).all() for key in ("poses", "trans", "betas")):
                raise ValueError("Non-finite body parameters: " + str(path))
            if path.stat().st_size != entry["size"]:
                raise ValueError("Source size changed since the download manifest: " + str(path))
            asset_family = canonical_asset_family(record["object_path"])
            split = family_split(asset_family)
            family_categories[asset_family].add(category)
            family_splits[asset_family] = split
            source_objects.add(record["object_path"])
            source_scale = float(np.asarray(human.get("scale", 1.)).reshape(-1)[0])
            obj_scale = np.asarray(obj.get("obj_scale", 1.)).reshape(-1).tolist()
            target_n = len(grail_source.output_clock(n, 24., 30.)[1])
            usd_relative = str(Path(entry["path"]).parent.parent / "object_usd" / (sequence + ".usd"))
            meta_relative = str(Path(entry["path"]).parent.parent / "meta" / path.name)
            dependencies = {}
            for name, relative in (("recon", entry["path"]), ("usd", usd_relative), ("meta", meta_relative)):
                if relative not in manifest_entries or not (source_root / relative).is_file():
                    raise ValueError("Missing source dependency: " + relative)
                identity = manifest_entries[relative]
                dependencies[name] = {key: identity.get(key) for key in ("size", "sha256", "blob_id")}
            row = dict(sequence_id=sequence, category=category,
                       recon_path=str(path), usd_path=str(source_root / usd_relative),
                       meta_path=str(source_root / meta_relative),
                       source_frames=n, source_fps=24., source_raw_fps=human.get("mocap_frame_rate"),
                       output_fps=30., output_frames=target_n,
                       source_duration_s=(n - 1) / 24., human_scale=source_scale,
                       obj_scale=obj_scale, model=human.get("model"), gender=human.get("gender"),
                       betas_nonzero=bool(np.any(np.abs(human["betas"]) > 1e-8)),
                       source_object_path=record["object_path"], object_family_id=asset_family,
                       split=split, source_root=str(source_root), source_identities=dependencies,
                       template_status="unresolved_human_template_provenance")
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            stats = category_stats.setdefault(category, dict(sequences=0, frames=Counter(), raw_fps=Counter(),
                human_scale=Counter(), object_scale=Counter(), model=Counter(), gender=Counter(),
                betas_nonzero=0, dynamic_objects=0, scene_type=Counter(), root_xy_displacements=[],
                root_z_deltas=[], families=set(), splits=Counter(), source_bytes=0))
            stats["sequences"] += 1
            stats["frames"][n] += 1
            stats["raw_fps"][str(human.get("mocap_frame_rate"))] += 1
            stats["human_scale"][str(source_scale)] += 1
            stats["object_scale"][str(obj_scale)] += 1
            stats["model"][str(human.get("model"))] += 1
            stats["gender"][str(human.get("gender"))] += 1
            stats["betas_nonzero"] += row["betas_nonzero"]
            stats["scene_type"][type(record.get("scene_data")).__name__] += 1
            stats["dynamic_objects"] += int(np.max(np.abs(obj["obj_t"] - obj["obj_t"][0])) > 1e-4 or
                                            np.max(np.abs(obj["obj_R"] - obj["obj_R"][0])) > 1e-4)
            trans = human["trans"]
            stats["root_xy_displacements"].append(float(np.linalg.norm(trans[-1, :2] - trans[0, :2])))
            stats["root_z_deltas"].append(float(trans[-1, 2] - trans[0, 2]))
            stats["families"].add(asset_family)
            stats["splits"][split] += 1
            stats["source_bytes"] += entry["size"]
            total_frames += n
            target_frames += target_n
            recon_bytes += entry["size"]
            if index % 2000 == 0:
                print(json.dumps(dict(scanned=index + 1, total=len(entries),
                                      elapsed_seconds=round(time.time() - started, 1))), flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "inventory.jsonl")
    for stats in category_stats.values():
        stats["asset_families"] = len(stats.pop("families"))
        for name in ("root_xy_displacements", "root_z_deltas"):
            values = stats.pop(name)
            stats[name + "_quantiles_m"] = dict(zip(("min", "p25", "median", "p75", "max"),
                                                     np.quantile(values, (0, .25, .5, .75, 1)).tolist()))
    summary = dict(protocol=PROTOCOL, source_root=str(source_root),
        repo=manifest["repo"], revision=manifest["revision"],
        manifest_sha256=grail_source.file_sha256(manifest_path),
        categories=category_stats, sequences=len(entries), source_recon_bytes=recon_bytes,
        source_frames=total_frames, resampled30_frames=target_frames,
        source_object_paths=len(source_objects), canonical_asset_families=len(family_categories),
        cross_category_asset_families={key: sorted(value) for key, value in family_categories.items() if len(value) > 1},
        asset_split_counts=dict(Counter(family_splits.values())),
        inventory_sha256=digest.hexdigest(), elapsed_seconds=time.time() - started,
        frame_rate_policy="Override stale per-record 30 with dataset-card/retargeting-source 24 Hz; retain both fields; resample to 30 Hz without extrapolation.",
        template_status="unresolved_human_template_provenance; inventory membership does not certify training quality",
        split_rule="grail_data.family_split(canonical_asset_family): deterministic 80/10/10 asset holdout before windows",
        checksum_scope="Source identities copied from the pinned manifest; downloader reports all hashes verified. This scan checks existence, recon byte size, schema, and finiteness; it does not rehash every source file.",
        no_gpu=True, source_modifications=False)
    _atomic(output / "SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), sequences=len(entries),
                          inventory_sha256=digest.hexdigest(), elapsed_seconds=summary["elapsed_seconds"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan(args.source_root, args.output)


if __name__ == "__main__":
    main()
