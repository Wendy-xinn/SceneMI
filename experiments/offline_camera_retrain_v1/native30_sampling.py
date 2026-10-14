"""Quota-balanced TRAIN draws and recording cycles, with resumable state."""
import copy
import hashlib
import json
from types import SimpleNamespace
import numpy as np

PROTOCOL = 'native30-recording-cycles-v1'


def recording_key(meta, group):
    dataset = 'egobody' if group in ('camera_wearer', 'interactee') else group
    return dataset + '/' + (meta.get('recording') or meta['sequence_id'])


def validate_sampling_options(options, ego_probability):
    if options.get('protocol') != PROTOCOL:
        raise ValueError('Unknown recording sampling protocol')
    block = options['mode_block_size']
    if isinstance(block, bool) or not isinstance(block, int) or block < 2:
        raise ValueError('Positive mode block size required')
    count = ego_probability * block
    single = options.get('training_mode')
    endpoint = (single == 'ego' and np.isclose(ego_probability,1,atol=1e-12,rtol=0)) or (single == 'exo' and np.isclose(ego_probability,0,atol=1e-12,rtol=0))
    if not np.isclose(count, round(count)) or (not endpoint and not 0 < round(count) < block):
        raise ValueError('Mode block must represent ego/exo ratio exactly')
    for key in ('recording_cooldown', 'recent_windows_per_target'):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Sampling history lengths must be positive integers')


class StratifiedRecordingSampler:
    def __init__(self, data, plan, options, seed):
        if data.split != 'train':
            raise ValueError('Training sampler cannot open VAL/test')
        self.data = data; self.plan = copy.deepcopy(plan); self.options = copy.deepcopy(options)
        self.ego_probability = sum(r['probability'] for r in plan if r['mode'] == 'ego')
        validate_sampling_options(options, self.ego_probability)
        self.rng = np.random.default_rng(seed)
        self.pools = {}
        for row in plan:
            bucket = row['group'] + '/' + row['mode']; pool = {}
            for i, (meta, _) in enumerate(data.members[row['group']]):
                starts = data.starts_for(row['group'], i, row['mode'])
                if row['mode'] not in meta.get('allowed_modes', ['ego','exo']) or not len(starts):
                    continue
                record = recording_key(meta, row['group'])
                pool.setdefault(record, []).extend((i, int(s)) for s in starts)
            if not pool:
                raise ValueError('Sampling quota has no eligible records: ' + bucket)
            self.pools[bucket] = pool
        fields=('sequence_id','recording','path','source_first','bundle_id','input_sha256','allowed_modes')
        sources={g:[{k:meta.get(k) for k in fields} for meta,_ in members] for g,members in data.members.items()}
        signature = dict(plan=plan, options=options, frames=data.frames, pools=self.pools, sources=sources)
        self.fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.deficits = {mode: [0.] * sum(r['mode'] == mode for r in plan) for mode in ('ego','exo')}
        self.mode_queue = []; self.record_queues = {}; self.window_queues = {}
        self.recent_records = []; self.recent_windows = {}; self.draws = 0

    @classmethod
    def for_bank(cls, identities, frames, plan, options, seed):
        members = {g: [] for g in ('trumans','camera_wearer','interactee','rich')}
        for i, identity in enumerate(identities):
            meta = dict(identity, bank_index=i, allowed_modes=[identity['mode']])
            members[identity['group']].append((meta, np.array([identity['start']])))
        def starts_for(group, i, mode):
            meta, starts = members[group][i]
            return starts if mode in meta['allowed_modes'] else np.empty(0, dtype=int)
        return cls(SimpleNamespace(split='train', frames=frames, members=members, starts_for=starts_for), plan, options, seed)

    def draw(self):
        if not self.mode_queue:
            block = self.options['mode_block_size']; n = round(self.ego_probability * block)
            self.mode_queue = self.rng.permutation(['ego'] * n + ['exo'] * (block-n)).tolist()
        mode = self.mode_queue.pop()
        rows = [r for r in self.plan if r['mode'] == mode]
        deficits = np.array(self.deficits[mode]) + np.array([r['probability'] for r in rows]) / (self.ego_probability if mode == 'ego' else 1-self.ego_probability)
        best = np.flatnonzero(np.isclose(deficits, deficits.max(), atol=1e-12, rtol=0))
        ix = int(self.rng.choice(best)); deficits[ix] -= 1; self.deficits[mode] = deficits.tolist()
        row = rows[ix]; bucket = row['group'] + '/' + mode; pool = self.pools[bucket]
        queue = self.record_queues.setdefault(bucket, [])
        if not queue:queue.extend(self.rng.permutation(sorted(pool)).tolist())
        available = [i for i, record in enumerate(queue) if record not in self.recent_records]
        cooldown_fallback = not available
        pick = available[0] if available else 0
        record = queue.pop(pick)
        key = bucket + '|' + record
        windows = self.window_queues.setdefault(key, [])
        if not windows:windows.extend(self.rng.permutation(len(pool[record])).tolist())
        # Across ego/exo, the same target/time uses the same overlap history;
        # the two EgoBody people are distinct targets of the same recording.
        target = record + '/' + row['group']
        history = self.recent_windows.setdefault(target, [])
        def interval(index):
            member, start = pool[record][index]
            meta, _ = self.data.members[row['group']][member]
            first = int(meta.get('source_first', 0)) + start
            return first, first+self.data.frames
        choices = [i for i, index in enumerate(windows) if all(interval(index)[1] <= lo or interval(index)[0] >= hi for lo, hi in history)]
        overlap_fallback = not choices
        chosen = windows.pop(choices[0] if choices else 0)
        member, start = pool[record][chosen]; history.append(interval(chosen))
        del history[:-self.options['recent_windows_per_target']]
        self.recent_records.append(record); del self.recent_records[:-self.options['recording_cooldown']]
        self.draws += 1
        return dict(group=row['group'], mode=mode, member=member, start=start,
                    recording_key=record, cooldown_fallback=cooldown_fallback, overlap_fallback=overlap_fallback)

    def state_dict(self):
        return copy.deepcopy(dict(protocol=PROTOCOL, fingerprint=self.fingerprint, options=self.options,
            plan=self.plan, rng=self.rng.bit_generator.state, deficits=self.deficits, mode_queue=self.mode_queue,
            record_queues=self.record_queues, window_queues=self.window_queues,
            recent_records=self.recent_records, recent_windows=self.recent_windows, draws=self.draws))

    def load_state_dict(self, state):
        if state['protocol'] != PROTOCOL or state['fingerprint'] != self.fingerprint:
            raise ValueError('Sampler configuration/source windows changed at resume')
        state = copy.deepcopy(state); self.rng.bit_generator.state = state['rng']
        for name in ('deficits','mode_queue','record_queues','window_queues','recent_records','recent_windows','draws'):
            setattr(self, name, state[name])
