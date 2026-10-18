"""Freeze recording-independent EXO VAL expansion without inspecting outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .validation_cohort import common_members, require, sha

BASE = Path(__file__).resolve().parents[1]
PRIOR = BASE/'runs/native30_exo_adapter_v1/prepared/validation_plan.json'
DESIGN = BASE/'runs/native30_exo_adapter_v1/diagnostics/EXPANDED_EXO_VALIDATION_PLAN.json'
CONFIG = BASE/'runs/native30_gait_v2_ego55k_oct15/config.json'
SEED = 2026101808
PROTOCOL = 'native30-expanded-exo-recording-cohort-v1'


def rank(*identity):
    return hashlib.sha256(json.dumps([SEED, *identity], separators=(',', ':')).encode()).hexdigest()


def case_seed(config, row):
    key = json.dumps([row['group'], 'exo', row['sequence_id'], row['source_first']], separators=(',', ':'))
    return (config['validation']['seed'] + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')) % 2**32


def select(entries, previous, config, targets=None):
    """Exact registered identity-only selection; no GT, position or prediction."""
    targets = targets or dict(rich=27, trumans=20, egobody=17)
    by_dataset = defaultdict(list)
    for row in entries:
        by_dataset[row['dataset']].append(row)
    identities = {(e['group'], e['sequence_id'], e['start']): e for e in entries}
    require(len(identities) == len(entries), 'Duplicate legal EXO identity')
    old = defaultdict(list)
    for prior in previous:
        if prior['mode'] != 'exo':
            continue
        key = prior['group'], prior['sequence_id'], prior['start']
        require(key in identities, 'Prior EXO regression source is no longer legal VAL')
        row = dict(identities[key], panel='old9', case_seed=prior['case_seed'])
        require(all(row[k] == prior[k] for k in ('dataset','recording','source_first')), 'Prior EXO identity changed')
        require(case_seed(config, row) == row['case_seed'], 'Prior EXO case seed changed')
        old[row['dataset']].append(row)
    selected = []
    for dataset in ('rich', 'trumans', 'egobody'):
        prior = old[dataset]; records = {r['recording'] for r in prior}
        require(len(records) == len(prior) == 3, 'Need three original distinct EXO regression recordings per dataset')
        all_records = {r['recording'] for r in by_dataset[dataset]}
        available = sorted(all_records-records, key=lambda name: (rank(dataset, name), name))
        needed = targets[dataset]-len(prior)
        require(0 <= needed <= len(available), 'Insufficient independent EXO VAL recordings')
        if dataset in ('rich','egobody'):
            require(needed == len(available), 'Registered RICH/EgoBody review includes every eligible recording')
        roles = Counter(row['group'] for row in prior)
        for name in available[:needed]:
            windows = [row for row in by_dataset[dataset] if row['recording'] == name]
            if dataset == 'egobody':
                role = min({row['group'] for row in windows}, key=lambda value: (roles[value], rank(dataset, name, value), value))
                windows = [row for row in windows if row['group'] == role]
            windows.sort(key=lambda row: (row['source_first'], row['sequence_id'], row['source_bundle_ordinal'], row['start']))
            row = dict(windows[len(windows)//2], panel='new55')
            row['case_seed'] = case_seed(config, row)
            selected.append(row); roles[row['group']] += 1
        selected.extend(prior)
    for index, row in enumerate(selected):
        row['case'] = index
    return selected


def build(config, *, prior_plan=PRIOR, design_path=DESIGN):
    import numpy as np
    require(config['frames'] == 192 and config['fps'] == 30 and config['window_stride'] == 64, 'Expected native192/30Hz/grid64')
    require(config['validation']['seed'] == 2026101351 and config['physical_guidance']['enabled'], 'Original seed/physics protocol differs')
    design = json.loads(Path(design_path).read_text())
    require(design['status'] == 'design_not_sampled' and design['exact_selection']['seed'] == SEED, 'Registered EXO design differs')
    manifest = (BASE/config['train_manifest']).resolve(); index_path = (BASE/config['window_index']).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    index = json.loads(index_path.read_text()); info = index['files']['validation/exo']; array_path = index_path.parent/info['file']
    require(index['source_manifest_sha256'] == sha(manifest) and index['bundle_paths'] == [row['path'] for row in rows], 'Manifest/index identity differs')
    require(index['frames'] == 192 and index['fps'] == 30 and index['window_stride'] == 64 and index['exo_max_empty_frames'] == config['exo_max_empty_frames'], 'Legal EXO index protocol differs')
    require(sha(array_path) == info['sha256'], 'EXO legal window index SHA differs')
    for path, digest in design['source_sha256'].items():
        require(sha(path) == digest, 'Registered design source changed: '+path)
    starts = np.load(array_path, allow_pickle=False)
    require(starts.dtype == np.int64 and starts.shape == (info['windows'],2), 'Invalid legal EXO grid')
    require(len({(int(i),int(s)) for i,s in starts}) == len(starts), 'Duplicate legal EXO windows')
    members, validity = common_members(rows,192,64)
    entries = []
    for ordinal, start in starts:
        require(0 <= ordinal < len(rows), 'Invalid source ordinal')
        meta = rows[int(ordinal)]
        require(meta['split'] == 'validation' and 'exo' in meta['allowed_modes'] and not meta['diagnostic_only'], 'Nonlegal VAL source')
        require(0 <= start and start+192 <= meta['frames'] and start % 64 == 0, 'Illegal source window')
        entries.append(dict(dataset=meta['dataset'],group=meta['group'],mode='exo',member=members[int(ordinal)],
            path=meta['path'],sequence_id=meta['sequence_id'],recording=meta.get('recording') or meta['sequence_id'],
            source_bundle_ordinal=int(ordinal),start=int(start),source_first=int(meta['source_first']+start),joints=None))
    selected = select(entries,json.loads(Path(prior_plan).read_text())['plan'],config)
    coverage = {}
    for dataset in ('rich','trumans','egobody'):
        chosen = [row for row in selected if row['dataset'] == dataset]
        coverage[dataset] = dict(windows=len(chosen),recordings=len({r['recording'] for r in chosen}),
            roles=dict(Counter(r['group'] for r in chosen)),panels=dict(Counter(r['panel'] for r in chosen)),
            eligible_windows=sum(r['dataset']==dataset for r in entries),
            eligible_recordings=len({r['recording'] for r in entries if r['dataset']==dataset}))
    require(len(selected) == 64 and dict(Counter(r['panel'] for r in selected)) == dict(new55=55,old9=9), 'Registered64-recording expansion differs')
    require(all(r['dataset']=='rich' and r['panel']=='new55' for r in selected[:24]), 'RICH new24 must execute first')
    sources = {str(Path(path).resolve()):sha(path) for path in (Path(__file__),prior_plan,design_path,manifest,index_path,array_path)}
    sources.update(validity)
    return dict(protocol=PROTOCOL,prepared_at=datetime.now().astimezone().isoformat(),status='cohort_prepared_not_predicted',
        split='validation',mode='exo',frames=192,fps=30,window_stride=64,seed=SEED,regimes=['clean','noisy'],
        entries=selected,coverage=coverage,windows=64,new_recordings=55,regression_recordings=9,
        predictions_per_model=128,source_sha256=sources,TEST_used=False,optimizer_updates=0,predictions_used_for_selection=False,
        execution_order='RICH new24 then old3; TRUMANS new17 then old3; EgoBody new14 then old3',
        selection='Exact prior registered identity/hash/upper-middle rules; one EXO window per dataset+recording. Prior9 unchanged; no EGO cases. EgoBody official VAL contains17 recordings total, not20.',
        reporting='Report new55, old9 regression and all64 separately; RICH new24/all27; clean/noisy, equal-recording within each dataset then equal-dataset macro. Subjects/scenes can repeat across recordings.')


def validate_entries(data, entries, config):
    """Bind every declared source/member/start to a real official VAL loader."""
    require(data.split == 'validation' and len(entries) == 64, 'Expected registered64 EXO VAL entries')
    seen = set(); counts = Counter(); panels = Counter(); roles = defaultdict(Counter)
    for index,row in enumerate(entries):
        require(row['case'] == index and row['mode'] == 'exo' and row['joints'] is None, 'Invalid EXO case')
        meta,_ = data.members[row['group']][row['member']]
        require(meta['split'] == 'validation' and not meta['diagnostic_only'] and 'exo' in meta['allowed_modes'], 'Only official EXO VAL is legal')
        require(all(row[key] == meta[key] for key in ('path','group','dataset','sequence_id')), 'VAL identity/member mismatch')
        require(row['recording'] == (meta.get('recording') or meta['sequence_id']), 'Raw recording mismatch')
        require(row['source_first'] == meta['source_first']+row['start'] and row['start'] in data.starts_for(row['group'],row['member'],'exo'), 'Source clock/legal grid mismatch')
        key = row['dataset'],row['recording']; require(key not in seen, 'Repeated EXO recording'); seen.add(key)
        require(row['case_seed'] == case_seed(config,row), 'Identity case seed mismatch')
        require(row['panel'] in ('new55','old9'), 'Unknown validation panel')
        counts[row['dataset']] += 1; panels[row['panel']] += 1; roles[row['dataset']][row['group']] += 1
    require(dict(counts) == dict(rich=27,trumans=20,egobody=17) and dict(panels) == dict(new55=55,old9=9), 'Registered recording/panel coverage differs')
    require(all(row['dataset']=='rich' and row['panel']=='new55' for row in entries[:24]), 'RICH new24 first')
    return dict(windows=dict(counts),recordings=dict(counts),panels=dict(panels),roles={k:dict(v) for k,v in roles.items()})


def prepare(out, *, config_path=CONFIG, prior_plan=PRIOR, design_path=DESIGN):
    out = Path(out); require(not (out/'COHORT.json').exists(), 'Existing immutable EXO cohort')
    config_path = Path(config_path)
    require(sha(config_path) == '89fadfc6d1af8cf0c819e503ac646443b6c8d281ce46abb5f7e8c53828a796ad', 'Use exact original41k inference config')
    result = build(json.loads(config_path.read_text()),prior_plan=prior_plan,design_path=design_path)
    result['config_sha256'] = sha(config_path)
    out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_bytes(config_path.read_bytes())
    (out/'COHORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare'])
    parser.add_argument('--out',type=Path,default=BASE/'runs/native30_exo_expanded_review_v1')
    args = parser.parse_args();result = prepare(args.out)
    print(json.dumps(dict(windows=result['windows'],coverage=result['coverage']),ensure_ascii=False))
