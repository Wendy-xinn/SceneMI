"""Common observed ground datum for all head-conditioned RICH methods."""
import numpy as np
from scenemi.data.heldout_geometry import HeldoutGeometryData
from scenemi.geometry.ground_frame import observed_ground_height,translate_sample_ground
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points


class GroundedHeldoutGeometryData(HeldoutGeometryData):
    def __init__(self,*args,ground_policy,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground_policy,self.ground_evidence = dict(ground_policy),{}

    def ground(self,row):
        if row['path'] not in self.ground_evidence:
            saved=self.native.cached(row)
            height,evidence=observed_ground_height(saved['static_points'],saved['static_normals'],
                saved['static_normal_valid'],self.ground_policy)
            self.ground_evidence[row['path']]=dict(height_m=height,evidence=evidence)
        return self.ground_evidence[row['path']]['height_m']

    def sample(self,source,rng,*,member=None,start=None):
        sample,identity=super().sample(source,rng,member=member,start=start)
        row=self.native.records[identity['member']]
        height=self.ground(row)
        sample=translate_sample_ground(sample,height)
        saved=self.native.cached(row)
        first=identity['start'];origin=np.asarray(saved['camera_position_scenemi_yup'][first],np.float64).copy()
        origin[1]=0.
        anchor=anchor_rotation(np.asarray(saved['camera_rotation_scenemi_yup'][first:first+128],np.float64))
        points=((saved['static_full']-origin)@anchor.T).astype(np.float32)
        points[:,1]-=height
        sample['occupancy']=occupancy_from_points(points)
        identity.update(ground_height_m=height,ground_source='observed surface geometry only',
            common_ground_datum_for_all_arms=True,ground_uses_target_motion=False)
        return sample,identity
