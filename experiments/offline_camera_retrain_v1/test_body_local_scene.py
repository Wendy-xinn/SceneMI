import numpy as np,torch
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneWindow

def main():
 q=np.array([0.,1.]);j=np.zeros((2,22,3));static=np.array([[.2,0,0],[.01,0,0]]);times=np.array([0.,1.]);empty=[np.empty((0,3))]*2
 a=BodySceneWindow(static,times,q,empty).features(j)
 b=BodySceneWindow(np.array([[.2,0,0],[.7,0,0]]),times,q,empty).features(j)
 np.testing.assert_array_equal(a[0],b[0]);assert a[1,0,3]<a[0,0,3]
 moving=BodySceneWindow(np.empty((0,3)),np.empty(0),q,[np.array([[.1,0,0]]),np.array([[2.,0,0]])]).features(j)
 assert moving[0,:,4].all() and moving[0,:,5].all();assert not moving[1].any()
 shifted=j.copy();shifted[:,:,0]=.5;c=BodySceneWindow(static,times,q,empty).features(shifted);assert not np.allclose(c,a)
 assert not BodySceneWindow(np.empty((0,3)),np.empty(0),q,empty).features(j).any()
 print('PASS future static cannot alter prefix; dynamic does not leave ghost; body motion changes query; unknown masked')
if __name__=='__main__':main()
