"""Fixed native body anchors, excluding unsupported fingers/jaw/eye regions."""
import numpy as np
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model


class Native30SkinAnchorCache(SkinAnchorCache):
    def get(self,info):
        key=(info['model'],info['gender'])
        if key not in self.ids:
            if key not in self.models:self.models[key]=load_model(*key).cpu()
            model=self.models[key];vertices=model.v_template.detach().numpy()
            dominant=model.lbs_weights.argmax(1).numpy();ids=[];labels=[]
            for joint in range(22):
                pool=np.flatnonzero(dominant==joint)
                if len(pool)<3:raise ValueError(f'Native body region {joint} has fewer than 3 anchors')
                # Feet/ankles focus on low support surfaces. Wrist markers
                # stay in the supported region instead of unmodelled fingers.
                if joint in (7,8,10,11):
                    low=pool[vertices[pool,1]<=np.quantile(vertices[pool,1],.3)]
                    if len(low)>=3:pool=low
                    first=int(np.argmin(vertices[pool,1]))
                else:
                    centre=vertices[pool].mean(0)
                    first=int(np.argmin(((vertices[pool]-centre)**2).sum(-1)))
                selected=[first];distance=np.full(len(pool),np.inf)
                for _ in range(2):
                    distance=np.minimum(distance,((vertices[pool]-vertices[pool[selected[-1]]])**2).sum(-1))
                    distance[selected]=-1;selected.append(int(distance.argmax()))
                ids.extend(pool[selected]);labels.extend([joint]*3)
            head=np.flatnonzero(dominant==15);extra=next(int(i) for i in head[np.argsort(vertices[head,2])[::-1]] if i not in ids)
            ids.append(extra);labels.append(15)
            self.ids[key]=np.array(ids),np.array(labels)
        return super().get(info)
