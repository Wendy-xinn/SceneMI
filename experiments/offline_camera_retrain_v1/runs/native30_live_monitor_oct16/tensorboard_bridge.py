#!/usr/bin/env python3
"""Read-only JSONL -> TensorBoard sidecar; never imports the training model.

Offsets, accumulated clocks and event batches commit in one SQLite transaction.
Each committed batch is published once as an immutable TFRecord event file via
atomic rename. A crash between commit and rename is recovered on the next poll;
uncommitted partial source lines are left unread. Only this monitor directory is
written. Missing historical absolute timestamps use the explicitly recorded import
time; source elapsed counters are preserved separately, never guessed from mtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import time

from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.custom_scalar import layout_pb2, metadata as custom_metadata
from tensorboard.summary.writer.record_writer import RecordWriter

from loss_accounting import native_contributions

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent


def dashboard_layout():
    """Show one total and three real weighted contributions before diagnostics."""
    return {
        '00 总损失': {
            'Total loss：实际反传的唯一总损失': [r'^loss/total$'],
        },
        '01 分项：已加权贡献，三项相加等于 Total loss': {
            '基础身体与步态 / 观测一致性 / 场景几何': [r'^loss/weighted/(base_body_and_gait|observation|scene_geometry)$'],
        },
        '02 固定验证：clean 与 noisy 分开': {
            '全身 MPJPE（mm）': [r'^validation/fixed/ego/(clean|noisy)/scene_on/mpjpe_mm/value$'],
            '头部位置误差（mm）': [r'^validation/fixed/ego/(clean|noisy)/scene_on/conditioned_mpjpe_mm/value$'],
            '身体坐标脚位误差（mm）': [r'^feet/fixed/(clean|noisy)/body_local_foot_error_mm$'],
            '脚速度 RMSE（m/s）': [r'^feet/fixed/(clean|noisy)/foot_velocity_RMSE_m_s$'],
            '步幅与左右脚关系误差': [r'^feet/fixed/(clean|noisy)/(articulation_log_ratio_error|bilateral_sagittal_correlation_error)$'],
        },
        '03 训练进度与真实计时': {
            '已完成训练步数': [r'^progress/optimizer_step$'],
            '学习率': [r'^train/lr$'],
            '裁剪前梯度范数': [r'^train/gradient_norm$'],
            '每步秒数：完整更新 / 计算 / 准备': [r'^time/(update|compute|preparation)_seconds_per_step$'],
            '累计小时：更新 / 计算 / 准备': [r'^time/train_(update|compute|preparation)_hours$'],
            '累计验证小时': [r'^time/validation_hours$'],
        },
        '90 详细：基础项原始日志，不在此再次相加': {
            'x0 与 raw_x0（raw 指未修正网络预测）': [r'^train/terms/base/(x0|raw_x0)$'],
            'FK位置误差（m2，尚未乘外层2）': [r'^train/terms/base/fk_position_m2$'],
            '根与头朝向原始误差': [r'^train/terms/base/(root_absolute_chordal|head_absolute_chordal)$'],
            '步态小计：final 与 raw（已含内部系数；raw另乘0.5）': [r'^train/terms/base/(gait_total|gait_raw_total)$'],
            '双脚幅度与步幅原始误差': [r'^train/terms/base/gait_(raw_)?bilateral_(stride|amplitude)$'],
            '诊断：旧脚速与planting，V2不直接加入total': [r'^train/terms/base/(foot_velocity_m2_s2|planting_m2_s2)$'],
        },
        '91 详细：观测各项已加权贡献，已包含于顶层观测分项': {
            '位置 / 旋转 / 线速度 / 线加速度 / 角速度 / 身体转动': [r'^loss/details/weighted_observation/.*$'],
        },
        '92 详细：场景各项已加权贡献，已包含于顶层场景分项': {
            '接触 / 距离 / 相对滑动 / 平面代理 / 对应 / 平滑 / 支撑': [r'^loss/details/weighted_scene/.*$'],
        },
        '93 详细：原始观测与几何，不是最终加权贡献': {
            '观测原始项（内部已有mask、归一化和噪声门控）': [r'^train/terms/observation/.*$'],
            '场景原始项（内部已有mask与门控）': [r'^train/terms/geometry/.*$'],
        },
        '94 详细：核对与轮换验证': {
            '三项加总减total：仅应有浮点舍入误差': [r'^loss/audit/weighted_sum_minus_total$'],
            '轮换全身误差（序列会变化，mm）': [r'^validation/rotating/ego/(clean|noisy)/scene_on/mpjpe_mm/value$'],
            '已记录循环累计小时（会话末验证可能漏计，下界）': [r'^time/active_loop_hours$'],
        },
        '95 旧55k：定义不同，不与新版总loss直接排名': {
            '旧模型总loss': [r'^train/losses/total$'],
            '旧步态 / 支撑 / 协调已加权小计': [r'^train/losses/(gait_total|phase_free_support_total|coordination_total)$'],
        },
    }


def dashboard_event():
    categories = []
    for title, charts in dashboard_layout().items():
        categories.append(layout_pb2.Category(title=title, closed=int(title[:2]) >= 90, chart=[
            layout_pb2.Chart(title=name, multiline=layout_pb2.MultilineChartContent(tag=tags))
            for name, tags in charts.items()]))
    layout = layout_pb2.Layout(category=categories)
    tensor = TensorProto(dtype='DT_STRING', string_val=[layout.SerializeToString()],
                         tensor_shape=TensorShapeProto())
    summary = Summary(value=[Summary.Value(tag=custom_metadata.CONFIG_SUMMARY_TAG,
                      tensor=tensor, metadata=custom_metadata.create_summary_metadata())])
    return Event(step=0, wall_time=time.time(), summary=summary)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten(value, prefix=''):
    """Preserve the source hierarchy and values, including already weighted totals."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f'{prefix}/{key}' if prefix else str(key))
    elif finite(value):
        yield prefix, float(value)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, f'{prefix}/{index}')


def timestamp(record, imported_at):
    for key in ('checked_at', 'timestamp', 'created_at'):
        text = record.get(key)
        if isinstance(text, str):
            try:
                parsed = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
                if parsed.tzinfo is not None:
                    return parsed.timestamp(), 'source_absolute'
            except ValueError:
                pass
        elif finite(text) and text > 1000000000:
            return float(text), 'source_absolute'
    return imported_at, 'import_time_no_source_absolute'


def scalar_event(step, walltime, values):
    summary = Summary(value=[Summary.Value(tag=key, simple_value=value)
                             for key, value in values.items() if finite(value)])
    return Event(step=int(step), wall_time=walltime, summary=summary)


def pack_events(events, generation_start=False):
    buffer = io.BytesIO()
    writer = RecordWriter(buffer)
    writer.write(Event(file_version='brain.Event:2').SerializeToString())
    if generation_start:
        writer.write(Event(step=0, wall_time=time.time(),
                           session_log=SessionLog(status=SessionLog.START)).SerializeToString())
    for event in events:
        writer.write(event.SerializeToString())
    return buffer.getvalue()


def initial_clock():
    return dict(previous_elapsed=None, elapsed_base=0., active_seconds=0.,
                update_seconds=0., compute_seconds=0., preparation_seconds=0.,
                validation_seconds=0., session=0, rows=0, latest_step=0)


def train_events(record, clock, imported_at, legacy=False):
    step = record['step']
    walltime, provenance = timestamp(record, imported_at)
    values = {}
    for key in ('loss', 'gradient_norm', 'lr', 'samples_seen', 'peak_gpu_gb',
                'length', 'support_cache_hits', 'support_cache_misses'):
        if finite(record.get(key)):
            values[f'train/{key}'] = float(record[key])
    for key in ('terms', 'losses'):
        for tag, value in flatten(record.get(key, {}), f'train/{key}'):
            values[tag] = value
    if isinstance(record.get('scene'), bool):
        values['train/scene_enabled'] = float(record['scene'])
    timesteps = record.get('timesteps', [])
    if timesteps and all(finite(v) for v in timesteps):
        values['train/diffusion_t_mean'] = sum(timesteps) / len(timesteps)
    elapsed = record.get('elapsed_s')
    if finite(elapsed):
        previous = clock.get('previous_elapsed')
        if previous is not None and elapsed < previous:
            clock['elapsed_base'] += previous
            clock['session'] += 1
        clock['previous_elapsed'] = elapsed
        clock['active_seconds'] = clock['elapsed_base'] + elapsed
        values['time/active_loop_hours'] = clock['active_seconds'] / 3600.
        values['time/active_loop_minutes'] = clock['active_seconds'] / 60.
        values['time/session_elapsed_seconds'] = elapsed
        values['time/session_index'] = float(clock['session'])
    for source_key, clock_key, tag in (
        ('update_s', 'update_seconds', 'update'),
        ('compute_s', 'compute_seconds', 'compute'),
        ('preparation_s', 'preparation_seconds', 'preparation')):
        if finite(record.get(source_key)):
            clock[clock_key] += record[source_key]
            values[f'time/{tag}_seconds_per_step'] = record[source_key]
            values[f'time/train_{tag}_hours'] = clock[clock_key] / 3600.
    values['progress/optimizer_step'] = float(step)
    values['provenance/walltime_is_source_absolute'] = float(provenance == 'source_absolute')
    clock['latest_step'] = max(clock['latest_step'], step)
    clock['rows'] += 1
    result = [('step', scalar_event(step, walltime, values))]
    total = record.get('loss', record.get('losses', {}).get('total'))
    if finite(total) and finite(elapsed):
        result.append(('active_seconds', scalar_event(int(clock['active_seconds']), walltime,
                      {'train/loss': total, 'progress/optimizer_step': float(step)})))
    return result


def validation_events(record, clock, imported_at, feet=False):
    step = record['step']
    walltime, _ = timestamp(record, imported_at)
    prefix = 'feet' if feet else 'validation'
    values = dict(flatten(record.get('summary', {}), prefix))
    # Per-case identity stays in the tag: rotating sets must not masquerade as a
    # single fixed sequence. Summary fixed/rotating curves remain separate above.
    for case in record.get('cases', []):
        identity = str(case.get('sequence_id', case.get('case', 'unknown')))
        identity = re.sub(r'[^A-Za-z0-9_.@-]', '_', identity)
        case_prefix = '/'.join([prefix + '_cases', str(case.get('panel', 'unknown')),
                                str(case.get('dataset', 'unknown')), identity,
                                str(case.get('regime', 'unknown'))])
        for tag, value in flatten(case.get('metrics', {}), case_prefix):
            values[tag] = value
    if finite(record.get('validation_seconds')):
        # Native foot_monitor owns the elapsed time for the whole validation
        # invocation. validation.jsonl has the metric panel, not a second timer.
        seconds = record['validation_seconds']
        clock['validation_seconds'] += seconds
        values['time/validation_seconds_this_call'] = seconds
        values['time/validation_hours'] = clock['validation_seconds'] / 3600.
    clock['latest_step'] = max(clock['latest_step'], step)
    clock['rows'] += 1
    return [('step', scalar_event(step, walltime, values))]


class SourceHistoryChanged(RuntimeError):
    pass


class Bridge:
    def __init__(self, directory, config):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.loss_coefficients = {}
        self.event_root = self.directory / 'tensorboard'
        self.event_root.mkdir(exist_ok=True)
        self.lock = (self.directory / '.bridge.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.open_database()

    def open_database(self):
        self.db = sqlite3.connect(self.directory / 'bridge.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS sources (
            key TEXT PRIMARY KEY, path TEXT NOT NULL, offset INTEGER NOT NULL,
            tail_sha TEXT, state_json TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT NOT NULL,
            payload BLOB NOT NULL, sha256 TEXT NOT NULL, published INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS images (
            path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', ('generation', str(time.time_ns())))
        self.db.commit()
        self.generation = int(self.db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])

    def close(self):
        self.db.close()
        self.lock.close()

    @staticmethod
    def fingerprint(handle, offset):
        position = handle.tell()
        handle.seek(max(0, offset - 4096))
        value = hashlib.sha256(handle.read(min(4096, offset))).hexdigest()
        handle.seek(position)
        return value

    def queue(self, run_name, events):
        if not events:
            return
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_name):
            raise ValueError(f'Unsafe run name: {run_name}')
        first_batch = self.db.execute('SELECT 1 FROM batches WHERE run_name=? LIMIT 1', (run_name,)).fetchone() is None
        payload = pack_events(events, generation_start=first_batch)
        self.db.execute('INSERT INTO batches(run_name,payload,sha256) VALUES(?,?,?)',
                        (run_name, payload, hashlib.sha256(payload).hexdigest()))

    def ingest_source(self, source, limit=250):
        path = Path(source['path']).resolve()
        if not path.exists():
            return 0
        key = source['key']
        stored = self.db.execute('SELECT path,offset,tail_sha,state_json FROM sources WHERE key=?',
                                 (key,)).fetchone()
        if stored:
            stored_path, offset, old_sha, state_json = stored
            if stored_path != str(path):
                raise RuntimeError(f'Source path changed for {key}; use a new source key')
            state = json.loads(state_json)
        else:
            offset, old_sha, state = 0, None, initial_clock()
        if path.stat().st_size < offset:
            raise SourceHistoryChanged(f'Source truncated: {path}; rebuilding from current source history')
        grouped = {}
        count = 0
        with path.open('rb') as handle:
            if old_sha and self.fingerprint(handle, offset) != old_sha:
                raise SourceHistoryChanged(f'Previously imported source bytes changed: {path}')
            handle.seek(offset)
            for _ in range(limit):
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b'\n'):
                    handle.seek(start)
                    break
                if not line.strip():
                    offset = handle.tell()
                    continue
                record = json.loads(line)
                offset = handle.tell()
                # A legacy log also contains evaluation completion rows without
                # losses. They are not training updates and carry no new clocks.
                if not isinstance(record.get('step'), int):
                    count += 1
                    continue
                role = source['role']
                if role == 'legacy_train' and not isinstance(record.get('losses'), dict):
                    count += 1
                    continue
                if role in ('native_trace', 'legacy_train'):
                    result = train_events(record, state, time.time(), role == 'legacy_train')
                elif role == 'native_loss_display':
                    effective_path = Path(source['effective_config']).resolve()
                    if str(effective_path) not in self.loss_coefficients:
                        effective = json.loads(effective_path.read_text())
                        self.loss_coefficients[str(effective_path)] = (
                            effective['observation_coefficients'],
                            effective['config']['geometry']['coefficients'])
                    obs, geom = self.loss_coefficients[str(effective_path)]
                    values = native_contributions(record, obs, geom)
                    result = [('step', scalar_event(record['step'], timestamp(record, time.time())[0], values))]
                    state['rows'] += 1
                    state['latest_step'] = max(state['latest_step'], record['step'])
                elif role in ('native_validation', 'native_feet'):
                    result = validation_events(record, state, time.time(), role == 'native_feet')
                else:
                    raise ValueError(f'Unknown source role: {role}')
                for axis, event in result:
                    run = source['run'] + ('__active_seconds' if axis == 'active_seconds' else '')
                    grouped.setdefault(run, []).append(event)
                count += 1
            tail_sha = self.fingerprint(handle, offset)
        # State and all event bytes become durable together, before publication.
        with self.db:
            for run_name, events in grouped.items():
                self.queue(run_name, events)
            self.db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)',
                            (key, str(path), offset, tail_sha, json.dumps(state)))
        return count

    def ingest_images(self):
        from PIL import Image
        count = 0
        for manifest_path in sorted((self.directory / 'visuals').glob('step_*/MANIFEST.json')):
            data = json.loads(manifest_path.read_text())
            step = data['step']
            walltime, _ = timestamp(data, time.time())
            for case in data.get('cases', []):
                for regime, supplied_path in case.get('images', {}).items():
                    path = Path(supplied_path)
                    if not path.is_absolute():
                        path = manifest_path.parent / path
                    path = path.resolve()
                    png = path.read_bytes()
                    sha = hashlib.sha256(png).hexdigest()
                    stored = self.db.execute('SELECT sha256 FROM images WHERE path=?',
                                             (str(path),)).fetchone()
                    if stored:
                        if stored[0] != sha:
                            raise RuntimeError(f'Imported image changed: {path}')
                        continue
                    with Image.open(io.BytesIO(png)) as im:
                        width, height = im.size
                    tag = '/'.join(['visual', str(case.get('panel', 'fixed')),
                                    str(case.get('dataset', 'unknown')), str(regime)])
                    summary = Summary(value=[Summary.Value(tag=tag, image=Summary.Image(
                        height=height, width=width, colorspace=3, encoded_image_string=png))])
                    with self.db:
                        self.queue(self.config.get('image_run', 'new_native30'),
                                   [Event(step=step, wall_time=walltime, summary=summary)])
                        self.db.execute('INSERT INTO images VALUES(?,?)', (str(path), sha))
                    count += 1
        return count

    def publish(self):
        rows = self.db.execute('SELECT id,run_name,payload,sha256 FROM batches WHERE published=0').fetchall()
        for batch_id, run_name, payload, sha in rows:
            run_dir = self.event_root / run_name
            run_dir.mkdir(exist_ok=True)
            destination = run_dir / f'events.out.tfevents.{self.generation:020d}.{batch_id:012d}.bridge'
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
                    raise RuntimeError(f'Published event batch changed: {destination}')
            else:
                temporary = run_dir / f'.pending-{batch_id:012d}'
                with temporary.open('wb') as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                directory_fd = os.open(run_dir, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            # If killed here, the existing file is verified and marked next time.
            with self.db:
                self.db.execute('UPDATE batches SET published=1 WHERE id=?', (batch_id,))
        return len(rows)

    def rebuild(self, reason):
        """A trainer rollback invalidates imported history; never splice its tail.

        Old immutable event files must remain readable for an already-running
        DirectoryWatcher to advance past its open file. New filenames sort after
        them, and the first new event uses SessionLog START at 0: both fresh and
        running readers purge every stale point before reloading current history.
        """
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
        retired = self.directory / 'history' / str(self.generation)
        retired.mkdir(parents=True, exist_ok=False)
        for name in ('bridge.sqlite3', 'bridge.sqlite3-wal', 'bridge.sqlite3-shm'):
            original = self.directory / name
            if original.exists():
                os.replace(original, retired / name)
        self.open_database()
        record = dict(checked_at=dt.datetime.now().astimezone().isoformat(), reason=reason,
                      retired_generation=str(retired), new_generation=self.generation,
                      policy='all sources and clocks rebuilt from current complete rows; SessionLog START(step=0) purges stale points; old immutable files stay readable so running TensorBoard advances safely')
        with (self.directory / 'REBUILD_EVENTS.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def status(self):
        sources = {}
        for key, path, offset, state in self.db.execute('SELECT key,path,offset,state_json FROM sources'):
            sources[key] = dict(path=path, offset=offset, **json.loads(state))
        result = dict(status='monitoring', checked_at=dt.datetime.now().astimezone().isoformat(),
                      pid=os.getpid(), sources=sources,
                      event_batches=self.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0],
                      generation=self.generation,
                      images=self.db.execute('SELECT COUNT(*) FROM images').fetchone()[0],
                      event_root=str(self.event_root),
                      historical_walltime='source checked_at when present; otherwise event import time, not actual training walltime',
                      timing='active_loop sums recorded elapsed sessions and is a lower bound: validation/save after the final trace row of a session may be absent; update/compute/preparation sum actual per-step source timers; validation is separately accumulated from foot_monitor; no pauses or setup time guessed')
        temporary = self.directory / '.BRIDGE_STATUS.pending'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, self.directory / 'BRIDGE_STATUS.json')
        return result

    def ensure_dashboard(self):
        key, version = 'dashboard_version', '2'
        stored = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        if stored and stored[0] == version:
            return
        with self.db:
            self.queue(self.config.get('image_run', 'new_native30'), [dashboard_event()])
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, version))
        (self.directory / 'DASHBOARD_LAYOUT.json').write_text(
            json.dumps(dashboard_layout(), ensure_ascii=False, indent=2) + '\n')

    def poll(self, limit=250):
        self.ensure_dashboard()
        count = 0
        try:
            for source in self.config['sources']:
                count += self.ingest_source(source, limit)
        except SourceHistoryChanged as error:
            self.rebuild(str(error))
            for source in self.config['sources']:
                count += self.ingest_source(source, limit)
        count += self.ingest_images()
        self.publish()
        return count, self.status()


def default_config():
    native = RUNS / 'native30_gait_v2_ego55k_oct15' / 'training'
    legacy = RUNS / 'native_dynamic_scene20_contact_55k_oct07'
    return dict(image_run='new_native30', sources=[
        dict(key='new_train', run='new_native30', role='native_trace', path=str(native / 'trace.jsonl')),
        dict(key='new_validation', run='new_native30', role='native_validation', path=str(native / 'validation.jsonl')),
        dict(key='new_feet', run='new_native30', role='native_feet', path=str(native / 'foot_monitor.jsonl')),
        dict(key='old_train', run='old55k', role='legacy_train', path=str(legacy / 'training_log.jsonl')),
        dict(key='new_loss_display_v1', run='new_native30', role='native_loss_display',
             path=str(native / 'trace.jsonl'),
             effective_config=str(native.parent / 'prepared' / 'effective_config.json')),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else default_config()
    bridge = Bridge(args.directory, config)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            count, status = bridge.poll()
            if count:
                print(json.dumps(dict(checked_at=status['checked_at'], imported=count,
                                      event_batches=status['event_batches'], images=status['images'])), flush=True)
            if not count:
                if args.once:
                    break
                time.sleep(args.poll_seconds)
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
