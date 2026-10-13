"""Official EgoBody recording splits; historical scopes are provenance only."""
import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache

from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS, SOURCE

SPLIT_PROTOCOL = 'egobody-official-recording-v2'
LEGACY_SPLIT_PROTOCOL = 'legacy-internal-scene-v1'
OFFICIAL_CSV = PROJECTS / 'egobody/data_splits.csv'
SPLITS = ('train', 'validation', 'test')


def normalize_split(split):
    split = 'validation' if split == 'val' else split
    if split not in SPLITS:
        raise ValueError(f'Unknown split: {split}')
    return split


def official_recordings(path=OFFICIAL_CSV):
    result = {}
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['train', 'val', 'test']:
            raise ValueError('Unexpected official EgoBody split columns')
        for row in reader:
            for split, recording in row.items():
                recording = recording.strip()
                if recording:
                    if recording in result:
                        raise ValueError(f'Repeated official recording: {recording}')
                    result[recording] = normalize_split(split)
    return result


@lru_cache(maxsize=2)
def egobody_sources(csv_sha256):
    # The hash keys the cache; re-read official assignments after any CSV edit.
    official = official_recordings()
    result = []
    seen = set()
    for legacy_split in SPLITS:
        for line in (SOURCE / f'egobody_sequences/{legacy_split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            identity = row['sequence_id']
            if identity in seen:
                raise ValueError(f'Duplicate EgoBody scoped source: {identity}')
            seen.add(identity)
            recording = row['recording']
            if recording not in official:
                raise ValueError(f'Unlisted EgoBody recording: {recording}')
            result.append(dict(row, split=official[recording],
                legacy_split=legacy_split, pose_profile_split=legacy_split,
                allowed_modes=['ego'],
                split_protocol=SPLIT_PROTOCOL, official_split_csv_sha256=csv_sha256))
    # External input has its own full recording clock: never split it by PV
    # visibility/gaps. Ego PV segments and full master streams are disjoint modes.
    for line in (SOURCE/'sequences.jsonl').read_text().splitlines():
        source=json.loads(line)
        if source['dataset']!='egobody':continue
        recording,role=source['recording_id'],source['role']
        if recording not in official:raise ValueError('Raw EgoBody recording missing from official split')
        first,last=source['source_first_frame'],source['source_last_frame_inclusive']+1
        result.append(dict(source,recording=recording,role=role,
            sequence_id=f'{recording}_{role}_{first}-{last}_master_full',split=official[recording],
            source_first_frame=first,source_last_frame_exclusive=last,
            allowed_modes=['exo'],pose_profile_split=None,legacy_split=None,
            split_protocol=SPLIT_PROTOCOL,official_split_csv_sha256=csv_sha256,
            availability_reason='full synchronized master/body stream, independent of PV scopes'))
    return result


def official_sources():
    return egobody_sources(hashlib.sha256(OFFICIAL_CSV.read_bytes()).hexdigest())


def validate_official_bundle(meta, split):
    """Do not allow a relabelled legacy/test bundle into official TRAIN."""
    if meta.get('split_protocol') != SPLIT_PROTOCOL:
        raise ValueError('Official training requires newly indexed official split bundles')
    if meta['dataset'] == 'egobody':
        if meta.get('official_split_csv_sha256') != hashlib.sha256(OFFICIAL_CSV.read_bytes()).hexdigest():
            raise ValueError('Official EgoBody split CSV changed or was not recorded')
        if official_recordings().get(meta.get('recording')) != normalize_split(split):
            raise ValueError('EgoBody recording assigned to the wrong official split')


def split_summary():
    official = official_recordings()
    sources = official_sources()
    return {split: dict(official_recordings=sum(v == split for v in official.values()),
        available_recordings=len({r['recording'] for r in sources if r['split'] == split}),
        available_segments=sum(r['split'] == split for r in sources),
        roles=dict(Counter(r['role'] for r in sources if r['split'] == split)),
        exo_only_segments=sum(r['split']==split and r['allowed_modes']==['exo'] for r in sources),
        missing_recordings=sorted(r for r, s in official.items() if s == split and
            r not in {x['recording'] for x in sources})) for split in SPLITS}
