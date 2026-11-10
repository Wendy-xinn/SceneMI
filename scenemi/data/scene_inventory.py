"""Report actual loader membership and fixed validation coverage.

Counts are derived from the same loaders as training, not directory names.
Legal windows overlap and must not be interpreted as independent examples.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_dataset import SceneDataset
from scenemi.scene_repair import settings
from scenemi.training import ensure_bounded, now


def covered_frames(starts, length=128):
    """Union of half-open windows on one sequence, never sum overlaps."""
    starts = np.unique(np.asarray(starts, dtype=np.int64))
    if not len(starts):
        return 0
    return int(length+np.minimum(np.diff(starts), length).sum())


def scene_counts(data, cases):
    result = {}
    for dataset in data.groups:
        records = [r for r in data.records if r['dataset'] == dataset]
        by_path = defaultdict(list)
        for r in records:
            by_path[r['path']].append(r)
        counts, category_frames = Counter(), defaultdict(Counter)
        for path, same in by_path.items():
            starts = np.unique(np.concatenate([r['starts'] for r in same]))
            meta = data.metadata(same[0])
            counts['unique_eligible_windows'] += len(starts)
            counts['unique_sequence_frames'] += int(meta['frames'])
            usable = covered_frames(starts)
            if usable > int(meta['frames']):
                raise ValueError('Eligible frame union exceeds sequence length')
            counts['eligible_unique_frames'] += usable
            category = meta.get('category', dataset)
            category_frames[category]['sequence_frames'] += int(meta['frames'])
            category_frames[category]['eligible_unique_frames'] += usable
        categories = Counter()
        for path, same in by_path.items():
            categories[data.metadata(same[0]).get('category', dataset)] += 1
        selected = [c for c in cases if c['dataset'] == dataset]
        selected_by_path = defaultdict(list)
        for c in selected:
            selected_by_path[data.records[c['member']]['path']].append(c['start'])
        result[dataset] = dict(loader_records=len(records), unique_sequences=len(by_path),
            independent_families=len(data.families[dataset]),
            eligible_window_entries=sum(len(r['starts']) for r in records),
            **dict(counts), categories=dict(categories), category_frames=dict(category_frames),
            sequence_hours=counts['unique_sequence_frames']/30/3600,
            eligible_hours=counts['eligible_unique_frames']/30/3600,
            synthetic_chair_records=sum(bool(r.get('synthetic_chair')) for r in records),
            evaluated_windows=len(selected),
            evaluated_sequences=len({data.records[c['member']]['path'] for c in selected}),
            evaluated_families=len({c['family'] for c in selected}),
            evaluated_frame_entries=128*len(selected),
            evaluated_unique_frames=sum(covered_frames(s) for s in selected_by_path.values()))
    return result


def amass_counts(index):
    result = {}
    for split in ('train', 'val', 'test'):
        families = [f for f in index['families'] if f['split'] == split and f['eligible_128']]
        variants = [v for f in families for v in f['variants'] if v['eligible_128']]
        original = [v for v in variants if not v['reflected']]
        source_frames = sum(v['frames_after_relative_pose_drop'] for v in original)
        variant_frames = sum(v['frames_after_relative_pose_drop'] for v in variants)
        result[split] = dict(independent_families=len(families), eligible_variants=len(variants),
            eligible_windows=sum(v['possible_windows_128'] for v in variants),
            native_variant_frames=variant_frames, original_frames=source_frames,
            original_hours=source_frames/30/3600, variant_hours=variant_frames/30/3600,
            corpora=dict(Counter(f['corpus'] for f in families)))
    return result


def scene_candidates(paths, known_metadata):
    """All prepared train/validation sources, before the EGO/quality screen."""
    groups = {split:{name:{} for name in ('egobody','trumans','grail')}
              for split in ('train','validation')}
    for line in paths['native_manifest'].read_text().splitlines():
        row = json.loads(line)
        if row['split'] in groups and row['dataset'] in ('egobody','trumans'):
            groups[row['split']][row['dataset']][row['path']] = row
    for line in paths['grail_manifest'].read_text().splitlines():
        row = json.loads(line)
        split = Path(row['path']).parent.name
        if split not in groups:
            continue
        if row['path'] in known_metadata:
            meta = known_metadata[row['path']]
        else:
            file = Path(row['path'])/'metadata.json'
            if sha256(file) != row['metadata_sha256']:
                raise ValueError('Candidate source metadata changed')
            meta = json.loads(file.read_text())
        if meta['split'] != split or meta['fps'] != 30:
            raise ValueError('Candidate source clock or split changed')
        groups[split]['grail'][row['path']] = meta
    result = {}
    for split,datasets in groups.items():
        result[split] = {}
        for name,by_path in datasets.items():
            frames = sum(int(m['frames']) for m in by_path.values())
            categories = defaultdict(Counter)
            for meta in by_path.values():
                c = categories[meta.get('category',name)]
                c['sequences'] += 1
                c['frames'] += int(meta['frames'])
            result[split][name] = dict(sequences=len(by_path), frames=frames,
                hours=frames/108000, categories=dict(categories),
                length_only_windows=sum(max(0,int(m['frames'])-127) for m in by_path.values()),
                ego_interface_sequences=sum('ego' in m.get('allowed_modes',['ego']) and
                    not m.get('diagnostic_only',False) and m['frames']>=128 for m in by_path.values()))
    return result


def amass_candidates(index, paths):
    ledger = paths['prepared_amass']/'manifest.jsonl'
    if sha256(ledger) != index['manifest_sha256']:
        raise ValueError('AMASS preprocessing manifest changed')
    rows = [json.loads(s) for s in ledger.read_text().splitlines()]
    result = {}
    for split in ('train','val','test'):
        selected = [r for r in rows if r['split']==split]
        raw_frames, raw_seconds = 0, 0.
        for r in selected:
            if 'raw_frames' in r:
                frames, fps = r['raw_frames'], r['source_fps']
            else:
                if split == 'test':
                    raise ValueError('Missing test metadata; never open test motion arrays')
                # The upstream filename filter did not decode these records.
                # Read only pose-array shape and scalar FPS, never pose values.
                file = paths['amass_raw']/r['source']
                stat = file.stat()
                if (stat.st_size,stat.st_mtime_ns)!=(r['inventory']['bytes'],r['inventory']['mtime_ns']):
                    raise ValueError('Raw candidate identity changed')
                with zipfile.ZipFile(file) as archive:
                    with archive.open('poses.npy') as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1,0):
                            shape,_,_ = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2,0):
                            shape,_,_ = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError('Unexpected NPZ header version')
                        frames = shape[0]
                    with archive.open('mocap_framerate.npy') as stream:
                        fps = float(np.lib.format.read_array(stream,allow_pickle=False))
            raw_frames += frames
            raw_seconds += frames/fps
        variants = [v for r in selected for v in r.get('variants',[]) if not v['reflected']]
        result[split] = dict(raw_sequences=len(selected), raw_frames=raw_frames,
            raw_hours=raw_seconds/3600, filename_excluded_sequences=sum(r['state']=='filename_excluded' for r in selected),
            prepared_sequences_before_128=len(variants),
            prepared_frames_before_128=sum(v['frames_after_relative_pose_drop'] for v in variants),
            source_fps='mixed; original frame counts are not 30Hz counts')
    return result


def inventory(scene_paths, output, amass_paths):
    config = settings(ROOT/'configs/scene_ego128_residual.yaml')
    parent = json.loads((ROOT/config['base']['contract']).read_text())
    cases = parent['validation_cases']
    result = dict(updated_at_utc=now(), frames=128, fps=30, test_arrays_opened=False,
        scene_training_sampling='uniform_dataset_then_family_then_record_then_window',
        scene_metrics='equal corpus; family means; GRAIL additionally equal action-category weight',
        validation_scope='All scene-validation source families; sampled windows, not all records or frames.',
        scene_manifest_contract_sha256=parent['sha256'],
        scene={}, validation_coverage=cases['coverage'])
    members, known_metadata = {}, {}
    for split in ('train', 'validation'):
        data = SceneDataset(scene_paths, np.zeros(3), split=split, config=config)
        result['scene'][split] = scene_counts(data, cases['comprehensive'] if split == 'validation' else [])
        known_metadata.update(data.metadata_cache)
        members[split] = {name:dict(families=set(data.families[name]),
            paths={r['path'] for r in data.records if r['dataset'] == name}) for name in data.groups}
    overlaps = {name:{kind:sorted(members['train'][name][kind] & members['validation'][name][kind])
                     for kind in ('families','paths')} for name in data.groups}
    if any(values for group in overlaps.values() for values in group.values()):
        raise ValueError('Training/validation loader members overlap: '+str(overlaps))
    result['train_validation_member_overlap'] = overlaps
    result['overlap_scope'] = 'Loader source-family IDs and sequence paths; not a claim of subject- or scene-disjoint splits.'
    result['scene_before_filter'] = scene_candidates(data.paths, known_metadata)
    result['before_filter_scope'] = 'All source paths in the prepared 30Hz train/validation manifests, before EGO-interface, minimum-window and contact/visibility selection; not the entire original-video collection.'
    result['expanded_validation'] = {name:dict(
        windows=group['evaluated_windows']+group['unique_sequences']-group['evaluated_sequences'],
        sequences=group['unique_sequences']) for name,group in result['scene']['validation'].items()}
    result['frame_count_scope'] = dict(scene_sequence_frames='Full length of each eligible sequence path, once per path; GRAIL can include frames excluded by the window filter.',
        eligible_unique_frames='Union of frames touched by eligible 128-frame windows, once per sequence path; no overlap multiplication.',
        amass_original='Prepared 30Hz frames after relative-pose first-frame drop, excluding mirrored augmentation.',
        hours='frame_count / 30 / 3600; nominal frame duration')
    index_path = ROOT/'runs/amass128/index/complete.json'
    amass_index = json.loads(index_path.read_text())
    result['amass'] = amass_counts(amass_index)
    result['amass_before_filter'] = amass_candidates(amass_index,load_paths(amass_paths))
    result['amass_index_sha256'] = sha256(index_path)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'DATASET_DISTRIBUTION.json', result)
    lines = ['# 当前训练数据与验证覆盖', '',
        '统计来自实际 SceneDataset 加载器和已封存 AMASS 索引。窗口长度 128 帧、30 Hz。', '',
        '1416 是原场景实验的固定评测清单：覆盖全部验证来源分组，但不是所有验证序列、帧或所有合法滑窗。它不包含 AMASS 验证集或独立测试集。', '',
        '| 数据集 | 划分 | 来源分组 | 唯一序列 | 合法 128 帧窗口（去重） | 原评测窗口 | 原覆盖序列 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for split, groups in result['scene'].items():
        for dataset in ('egobody', 'trumans', 'grail'):
            d = groups[dataset]
            lines.append(f'| {dataset} | {split} | {d["independent_families"]:,} | {d["unique_sequences"]:,} | {d["unique_eligible_windows"]:,} | {d["evaluated_windows"]:,} | {d["evaluated_sequences"]:,} |')
    lines += ['', '## 筛选前后（30 Hz 场景数据）', '',
        '筛选前指已预处理候选清单中该划分的全部序列，包含随后因 EGO 接口、窗口长度、场景可见性或动作接触质量而不采用的序列；不是原始视频库全集。保留序列完整帧数与最终可用帧数分开统计。', '',
        '| 数据集 | 划分 | 筛选前序列 | 筛选前总帧数 | 保留序列 | 保留序列完整帧 | 最终可用帧 | 帧保留率 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split,groups in result['scene'].items():
        for name,d in groups.items():
            before=result['scene_before_filter'][split][name]
            lines.append(f'| {name} | {split} | {before["sequences"]:,} | {before["frames"]:,} | {d["unique_sequences"]:,} | {d["unique_sequence_frames"]:,} | {d["eligible_unique_frames"]:,} | {d["eligible_unique_frames"]/before["frames"]:.2%} |')
    lines += ['', 'EgoBody 的候选减少主要来自 EGO 允许输入配置：训练 330 条中 175 条满足接口与长度限制，验证 87 条中 45 条满足；不能把其余全部解释为动作质量差。序列和帧数按导出路径去重，同一录制的不同人物/传感器配置仍可能分别计数；样本时长不是独立拍摄时长。', '']
    lines += ['', '## 帧数与时长', '',
        '完整序列帧数按当前加载的序列路径去重；可用帧数为合法 128 帧窗口覆盖的帧并集，重叠不重复累计。GRAIL 筛掉的窗口使两者不同；均为 30 Hz，时长按帧数/30 计算。', '',
        '| 数据集 | 划分 | 完整序列帧数 | 筛选后可用帧数 | 可用时长（小时） |',
        '| --- | --- | ---: | ---: | ---: |']
    for split,groups in result['scene'].items():
        for dataset,d in groups.items():
            lines.append(f'| {dataset} | {split} | {d["unique_sequence_frames"]:,} | {d["eligible_unique_frames"]:,} | {d["eligible_hours"]:.3f} |')
        total_frames=sum(d['unique_sequence_frames'] for d in groups.values())
        total_usable=sum(d['eligible_unique_frames'] for d in groups.values())
        lines.append(f'| **合计** | **{split}** | **{total_frames:,}** | **{total_usable:,}** | **{total_usable/108000:.3f}** |')
    lines += ['', '原生 EgoBody/TRUMANS 每个加载序列抽取中间窗口及远端窗口；GRAIL 每个物体家族抽取两条序列的窗口（不足两条时用同序列远端窗口）。清单详见基座 CONTRACT.json 的 validation_cases。', '',
        '新实验保留原清单，为此前未覆盖的每条验证序列各补一个中间合法窗口：', '',
        '| 数据集 | 扩展评测窗口 | 覆盖验证序列 |', '| --- | ---: | ---: |']
    for name,group in result['expanded_validation'].items():
        lines.append(f'| {name} | {group["windows"]:,} | {group["sequences"]:,} |')
    lines += ['', '合计 2115 窗口、1741 条验证序列，仍不是穷举全部 361750 个合法滑窗。训练/验证来源分组 ID 与序列路径交集均为空；这不等于已经证明人物或所有场景完全独立。', '',
        '训练按数据集等权抽样，再在各数据集内按来源分组、序列和窗口抽样。验证先按来源分组聚合，GRAIL 另按动作类别等权，最终三个数据集等权；不是将所有窗口直接平均。', '',
        'GRAIL 训练使用已筛选 EGO 窗口，另含 400 条人工补齐椅子的训练增强；验证不补椅子、不按接触/可见性筛选。RICH 不进入当前三数据集训练及选模。', '',
        '| GRAIL 动作类别 | 训练唯一序列 | 验证唯一序列 | 固定评测窗口 |', '| --- | ---: | ---: | ---: |']
    for category in sorted(result['validation_coverage']['comprehensive']['grail_category_windows']):
        lines.append('| '+category+' | '+' | '.join(str(v) for v in [
            result['scene']['train']['grail']['categories'].get(category, 0),
            result['scene']['validation']['grail']['categories'].get(category, 0),
            result['validation_coverage']['comprehensive']['grail_category_windows'][category]])+' |')
    lines += ['', '| GRAIL 动作类别 | 训练完整帧 | 训练可用帧 | 验证完整/可用帧 |', '| --- | ---: | ---: | ---: |']
    for category in sorted(result['scene']['validation']['grail']['categories']):
        t=result['scene']['train']['grail']['category_frames'][category]
        v=result['scene']['validation']['grail']['category_frames'][category]
        lines.append(f'| {category} | {t["sequence_frames"]:,} | {t["eligible_unique_frames"]:,} | {v["sequence_frames"]:,} |')
    lines += ['', '| GRAIL 动作类别 | 训练筛选前序列 | 训练筛选前帧 | 验证筛选前序列 | 验证筛选前帧 |', '| --- | ---: | ---: | ---: | ---: |']
    for category in sorted(result['scene_before_filter']['validation']['grail']['categories']):
        t=result['scene_before_filter']['train']['grail']['categories'][category]
        v=result['scene_before_filter']['validation']['grail']['categories'][category]
        lines.append(f'| {category} | {t["sequences"]:,} | {t["frames"]:,} | {v["sequences"]:,} | {v["frames"]:,} |')
    lines += ['', '## AMASS 预训练数据', '',
        '| 划分 | 独立动作家族 | 合格变体序列（含镜像） | 原动作帧数 | 原动作时长（小时） | 含镜像帧数 | 合法 128 帧窗口（含镜像） |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, d in result['amass'].items():
        lines.append(f'| {split} | {d["independent_families"]:,} | {d["eligible_variants"]:,} | {d["original_frames"]:,} | {d["original_hours"]:.3f} | {d["native_variant_frames"]:,} | {d["eligible_windows"]:,} |')
    lines += ['', '| AMASS 划分 | 原始候选序列 | 原始总帧（混合帧率） | 原始时长（小时） | 128 帧筛选前 30 Hz 序列 | 128 帧筛选前 30 Hz 帧数 | 最终 30 Hz 帧数（不含镜像） |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split,d in result['amass_before_filter'].items():
        lines.append(f'| {split} | {d["raw_sequences"]:,} | {d["raw_frames"]:,} | {d["raw_hours"]:.3f} | {d["prepared_sequences_before_128"]:,} | {d["prepared_frames_before_128"]:,} | {result["amass"][split]["original_frames"]:,} |')
    lines += ['', 'AMASS 原始帧率各不相同，原始总帧不能直接与 30 Hz 帧数比较保留率。预处理还包括上游文件排除、中间 80% 裁切、30 Hz 重采样和首帧移除；30 Hz 筛选前列已完成这些步骤，随后再要求可提供 128 帧。缺失的训练/验证原始帧数仅读取 NPZ 数组形状和帧率元数据补齐，不读取动作数值；test 仅使用已有预处理记录。', '']
    lines += ['', 'AMASS 使用 train 训练，val 选模，test 独立评测。这里统计 test 索引成员数量，没有打开 test 动作数组或运行测试。', '',
        'AMASS 帧数来自已处理的 30 Hz 动作，已去掉相对姿态所需的首帧；镜像是增强，不算新的独立录制。', '',
        '各滑窗高度重叠，不能将窗口总数当作独立样本量；AMASS 动作家族与 GRAIL 物体家族也不是相同统计单位。', '']
    (output/'DATASET_DISTRIBUTION_zh.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-paths', type=Path, default=ROOT/'configs/scene_paths.local.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/data_inventory')
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    args = parser.parse_args()
    ensure_bounded()
    result = inventory(args.scene_paths, args.output, args.paths)
    print(json.dumps({k:v for k,v in result.items() if k not in ('amass',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
