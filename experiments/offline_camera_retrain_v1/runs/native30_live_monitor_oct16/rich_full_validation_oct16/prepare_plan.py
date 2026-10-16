"""Enumerate every RICH entry in the sealed official validation/ego grid."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[2]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build():
    config_path = BASE/'runs/native30_gait_v2_ego55k_oct15/config.json'
    cfg = json.loads(config_path.read_text())
    manifest_path = BASE/cfg['train_manifest']
    manifest = [json.loads(x) for x in manifest_path.read_text().splitlines() if x]
    index_path = BASE/'data/native30_official_windows64_v1/dataset.json'
    index = json.loads(index_path.read_text())
    starts_path = index_path.parent/index['files']['validation/ego']['file']
    assert sha(manifest_path) == index['source_manifest_sha256']
    assert sha(starts_path) == index['files']['validation/ego']['sha256']
    assert index['bundle_paths'] == [x['path'] for x in manifest]
    assert cfg['frames'] == index['frames'] == 192
    assert cfg['window_stride'] == index['window_stride'] == 64
    starts = np.load(starts_path, allow_pickle=False)
    selected, sources = [], []
    for ordinal, meta in enumerate(manifest):
        if meta['dataset'] != 'rich' or meta['split'] != 'validation':
            continue
        assert not meta['diagnostic_only'] and meta['fps'] == 30 and 'ego' in meta['allowed_modes']
        path = Path(meta['path'])/'metadata.json'
        stored = json.loads(path.read_text())
        for k in ['sequence_id','group','split','frames','source_first','input_sha256','output_sha256']:
            assert meta[k] == stored[k]
        legal = starts[starts[:,0] == ordinal,1]
        assert not len(legal) or ((np.diff(legal)>0).all() and (legal%64 == 0).all())
        covered = np.zeros(meta['frames'], dtype=bool)
        for start in legal:
            start = int(start)
            assert start >= 0 and start+192 <= meta['frames']
            covered[start:start+192] = True
            source_first = meta['source_first']+start
            key = json.dumps([meta['group'],'ego',meta['sequence_id'],source_first],separators=(',',':'))
            seed = (2026101351+int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little'))%2**32
            selected.append(dict(dataset='rich',recording=meta.get('recording') or meta['sequence_id'],
                group=meta['group'],sequence_id=meta['sequence_id'],source_bundle_ordinal=ordinal,
                path=meta['path'],start=start,source_first=source_first,case_seed=seed,
                eligible_windows=len(legal),mode='ego',joints=[15],
                source_hashes=meta['output_sha256'],source_input_hashes=meta['input_sha256'],
                source_metadata_sha256=sha(path)))
        sources.append(dict(sequence_id=meta['sequence_id'],path=meta['path'],frames=meta['frames'],
            legal_windows=len(legal),covered_unique_frames=int(covered.sum()),
            uncovered_frames=int((~covered).sum()),reason_if_empty='source_shorter_than_192_frames' if meta['frames']<192 else None))
    selected.sort(key=lambda x:(x['recording'],x['sequence_id'],x['start']))
    for case, entry in enumerate(selected): entry['case'] = case
    assert len(selected) == 206 and len({(x['path'],x['start']) for x in selected}) == 206
    assert len({x['recording'] for x in selected}) == 27 and len(sources) == 28
    return dict(protocol='native30-full-rich-official-validation-ego-grid-v1',
        prepared_at=datetime.now().astimezone().isoformat(),selection=dict(
            method='Every RICH pair from the persisted official validation/ego index, sorted by recording then start.',
            outcome_selection=False,all_indexed_RICH_windows=True,tail_append=False,padding=False,
            unique_recordings=27,source_sequences=28,
            note='One source has 165 frames and no legal 192-frame window; 856 tail frames of the other sources are outside this grid. See source_coverage.'),
        frames=192,fps=30,window_stride=64,mode='ego',joints=[15],regimes=['clean','noisy'],
        scene='on',ddim_steps=20,case_seed_base=2026101351,unique_windows=206,expected_sampling_calls=412,
        source_manifest=str(manifest_path),window_index=str(index_path),source_coverage=sources,
        covered_unique_frames=sum(x['covered_unique_frames'] for x in sources),
        window_frame_occurrences=206*192,
        source_sha256={str(x):sha(x) for x in [Path(__file__),config_path,manifest_path,index_path,starts_path]},
        aggregation='Full indexed window-weighted and equal-recording macro, clean/noisy separately. Overlapping windows are retained; not an independent TEST or an assumed external-paper protocol.',
        plan=selected)

if __name__ == '__main__':
    plan = build()
    target = OUT/'PLAN.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');tmp.replace(target)
    print(json.dumps(dict(windows=len(plan['plan']),calls=plan['expected_sampling_calls'],recordings=27,covered_frames=plan['covered_unique_frames'])))
