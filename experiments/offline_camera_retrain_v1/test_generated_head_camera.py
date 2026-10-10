"""Fixed-mount reconstruction must preserve errors and be equivariant in world frame."""
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.generated_head_camera import apply_camera_mount,diagnostic_fixed_mount,camera_errors,frustum_lines

def main():
    head=np.array([[1.,2.,3.],[2.,2.,3.]])
    rotation=Rotation.from_euler('y',[0,90],degrees=True).as_matrix();mount_r=Rotation.from_euler('x',10,degrees=True).as_matrix();mount_t=np.array([.01,.04,.1])
    camera,cr=apply_camera_mount(head,rotation,mount_r,mount_t)
    rr,tt=diagnostic_fixed_mount(head,rotation,camera,cr);assert np.allclose(rr,mount_r) and np.allclose(tt,mount_t)
    reconstructed,r=apply_camera_mount(head,rotation,rr,tt);assert np.allclose(reconstructed,camera) and np.allclose(r,cr)
    displaced,_=apply_camera_mount(head+np.array([.1,0,0]),rotation,rr,tt);assert np.isclose(camera_errors(displaced,r,camera,cr)['position_mean_cm'],10)
    q=Rotation.from_euler('xyz',[10,20,30],degrees=True).as_matrix();shift=np.array([3.,-1.,2.]);transformed,rot=apply_camera_mount(head@q.T+shift,q@rotation,mount_r,mount_t)
    assert np.allclose(transformed,camera@q.T+shift) and np.allclose(rot,q@cr)
    lines=frustum_lines(camera[0],cr[0]);assert lines.shape==(8,2,3) and np.linalg.norm(lines-camera[0],axis=-1).max()<.2
    print('PASS fixed mount roundtrip, generated displacement retained, world-frame equivariance, small frustum')
if __name__=='__main__':main()
