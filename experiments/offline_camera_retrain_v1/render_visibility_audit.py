"""Reproducible offline evidence: ego owner video plus mesh contact sheet."""
import argparse,json
from pathlib import Path
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from view_scene_visibility_v2 import owner_rgb


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clip',type=Path,required=True);a=parser.parse_args();p=a.clip
    def load(n):return np.load(p/(n+'.npy'),mmap_mode='r')
    meta=json.loads((p/'metadata.json').read_text());v=load('body_vertices_scenemi_yup');faces=load('body_faces');owner=load('owner');pos=load('camera_position_scenemi_yup');rot=load('camera_rotation_scenemi_yup');scene=load('scene_points_scenemi_yup')
    with imageio.get_writer(p/'ego_occlusion.mp4',fps=meta['output_fps'],macro_block_size=1) as writer:
        for o in owner:writer.append_data(np.repeat(np.repeat(owner_rgb(o),4,0),4,1))
    frames=[0,len(v)//2,len(v)-1]
    fig=plt.figure(figsize=(15,9))
    for k,t in enumerate(frames):
        ax=fig.add_subplot(2,3,k+1,projection='3d')
        def mesh(verts,f,c):
            poly=Poly3DCollection(verts[f][...,[0,2,1]],facecolor=c,edgecolor='none',alpha=.95)
            ax.add_collection3d(poly)
        mesh(v[t],faces,'#309bdd')
        for i,item in enumerate(meta.get('objects',[])):
            if 'chair' in item['name']:mesh(load(f'object_{i}_vertices')[t],load(f'object_{i}_faces'),'#edac35')
        other=p/'occluder_body_vertices_scenemi_yup.npy'
        if other.exists():
            vv=np.load(other,mmap_mode='r');nv=v.shape[1]
            for j in range(1,vv.shape[1]//nv):mesh(vv[t,j*nv:(j+1)*nv],faces,'#d0419b')
        center=v[t].mean(0);near=scene[np.linalg.norm(scene-center,axis=1)<3][::5]
        ax.scatter(near[:,0],near[:,2],near[:,1],s=.3,c='#bbbbbb',alpha=.3)
        cam=pos[t];forward=rot[t,:,2];ax.plot(*np.stack((cam,cam+.5*forward))[:,[0,2,1]].T,color='red',lw=3)
        ax.set_xlim(center[0]-1.4,center[0]+1.4);ax.set_ylim(center[2]-1.4,center[2]+1.4);ax.set_zlim(center[1]-1.2,center[1]+1.2);ax.set_box_aspect((1,1,1));ax.view_init(15,35)
        ax.set_title(f'Frame {t}');ax.set_xlabel('X');ax.set_ylabel('Z');ax.set_zlabel('Y')
        ax2=fig.add_subplot(2,3,k+4);ax2.imshow(owner_rgb(owner[t]));ax2.axis('off');ax2.set_title(f'Ego: self={(owner[t]==100).sum()}, other={(owner[t]>100).sum()}')
    fig.suptitle(meta['sequence_id']+' | gray=static yellow=objects blue=self magenta=other')
    fig.tight_layout();fig.savefig(p/'audit_contact_sheet.png',dpi=120);plt.close(fig)
    print(p/'audit_contact_sheet.png')


if __name__=='__main__':main()
