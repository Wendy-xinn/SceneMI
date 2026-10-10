"""Build BPS for planned cameras from a known static map and timed dynamics.

This optional deployment adapter does not change dataset caches. World dynamic
observations/forecasts belong to exactly one requested timestamp. Historical
object positions are never appended to static memory.
"""
import numpy as np
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import temporal_bps


def known_scene_bps(static_points, dynamic_points_by_time, camera_position, camera_rotation,
                    anchors, *, dynamic_known, static_first_observed=None,
                    decision_source_frame=None):
    """All coordinates world-y-up metres; rotations camera-local to world.

    static_points is an explicitly known static map. If it comes from a recording
    memory with first-observation times, both those times and the decision time
    must be provided: later observations are excluded. A surveyed map may omit
    them. dynamic_known marks externally supplied observations/forecasts per
    query time, NOT availability inferred from validation ground truth. Unknown
    dynamics contribute no geometry; this does not assert free space.

    Returned validity is the existing nearest-known-surface availability mask,
    not a complete per-object uncertainty/occupancy representation.
    """
    static=np.asarray(static_points,np.float32)
    pos=np.asarray(camera_position,np.float32);rot=np.asarray(camera_rotation,np.float32)
    probes=np.asarray(anchors,np.float32);known=np.asarray(dynamic_known,bool)
    n=len(pos)
    if static.ndim!=2 or static.shape[1]!=3 or pos.shape!=(n,3) or rot.shape!=(n,3,3) or probes.ndim!=2 or probes.shape[1]!=3 or known.shape!=(n,) or len(dynamic_points_by_time)!=n:
        raise ValueError('Invalid static/dynamic/camera shapes')
    if (static_first_observed is None)!=(decision_source_frame is None):
        raise ValueError('Recording static memory requires BOTH timestamps and decision time')
    if not all(np.isfinite(a).all() for a in (static,pos,rot,probes)):
        raise ValueError('Nonfinite known scene')
    if not np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(rot),1,atol=1e-4):
        raise ValueError('Camera rotations must be proper SO(3)')
    if static_first_observed is not None:
        timestamps=np.asarray(static_first_observed)
        if timestamps.shape!=(len(static),) or not np.isfinite(timestamps).all() or not np.isfinite(decision_source_frame):
            raise ValueError('Invalid static observation timestamps')
        static=static[timestamps<=decision_source_frame]
    empty=np.empty((0,3),np.float32)
    dynamic=[]
    for available,points in zip(known,dynamic_points_by_time):
        if not available:
            dynamic.append(empty);continue
        points=np.asarray(points,np.float32)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Known dynamic surfaces must be finite [N,3]')
        dynamic.append(points)
    bps,valid=temporal_bps([empty]*n,dynamic,pos,rot,probes,return_valid=True,initial_static=static)
    return dict(bps=bps,bps_valid=valid,dynamic_known=known.copy(),static_points_available=len(static))
