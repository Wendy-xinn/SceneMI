"""Export report charts, group-specific results, and sequence-level CSVs."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.offline_camera_retrain_v1.evaluate_report_oct07 import aggregate
HERE=Path(__file__).parent;OUT=HERE/'runs/report55k_fullval_oct07'

def main():
    stats=json.loads((OUT/'dataset_stats.json').read_text());rows=[json.loads(l) for l in (OUT/'rows.jsonl').read_text().splitlines() if l];status=json.loads((OUT/'status.json').read_text())
    groups=aggregate(rows)
    colours=['#178a70','#db6654'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(11,4.3));x=np.arange(3);names=['TRUMANS','EgoBody','RICH']
    for i,(split,title) in enumerate([('train','Usable training corpus'),('validation','Usable validation corpus')]):
        count=[stats[split][g]['frames_20hz']/72000 for g in ['trumans','egobody','rich']]
        bars=ax[i].bar(x,count,color=['#316c99','#178a70','#db6654']);ax[i].set(xticks=x,xticklabels=names,ylabel='Person-hours (20 Hz frame count)',title=title)
        for b,g in zip(bars,['trumans','egobody','rich']):r=stats[split][g];ax[i].text(b.get_x()+b.get_width()/2,b.get_height(),f'{r["sequences"]} tracks\n{r["frames_20hz"]:,} frames',ha='center',va='bottom',fontsize=9)
        ax[i].set_ylim(0,max(count)*1.3)
    fig.suptitle('Actual experiment corpus; interpolated RICH frames are not independent observations');fig.tight_layout();fig.savefig(OUT/'dataset_size.png',dpi=180);plt.close(fig)
    if status['status']!='completed':print('dataset chart saved; waiting for complete metrics');return
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axes,['mpjpe_cm','wa_mpjpe_cm','support_floating_m'],['Global MPJPE (cm)','WA-MPJPE (cm)','GT support floating proxy (cm)']):
        for j,(variant,label) in enumerate([('scene_on','Scene on'),('scene_off','Scene off')]):
            vals=[groups[g]['variants'][variant]['sequence_mean'][key]*(100 if key=='support_floating_m' else 1) for g in ['trumans','egobody','rich']]
            bars=ax.bar(x+(j-.5)*.34,vals,width=.34,color=colours[j],label=label)
            for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height():.1f}',ha='center',va='bottom',fontsize=9)
        ax.set(xticks=x,xticklabels=names,title=title);ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    axes[0].legend();fig.suptitle('Old 55k: exhaustive eligible validation, equal weight per sequence, paired head control');fig.tight_layout();fig.savefig(OUT/'scene_ablation_metrics.png',dpi=180);plt.close(fig)
    # CSV preserves each sequence as one statistical unit.
    by_seq={}
    for row in rows:by_seq.setdefault((row['window']['group'],row['window']['sequence_id']),[]).append(row)
    keys=sorted(rows[0]['scene_on'])
    with (OUT/'sequence_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['group','sequence_id','windows','variant']+keys)
        for (group,sequence),items in sorted(by_seq.items()):
            for variant in ['scene_on','scene_off']:writer.writerow([group,sequence,len(items),variant]+[float(np.mean([r[variant][key] for r in items])) for key in keys])
    for name,members in [('TRUMANS',['trumans']),('EgoBody',['camera_wearer','interactee']),('RICH',['rich'])]:
        dest=OUT/'datasets'/name;dest.mkdir(parents=True,exist_ok=True);subset=[r for r in rows if r['window']['group'] in members]
        result={'full_coverage':True,'all_eligible':aggregate(subset)['all'],'primary_128':aggregate([r for r in subset if r['window']['length']==128])['all'],'short_64':aggregate([r for r in subset if r['window']['length']==64])['all']}
        (dest/'metrics.json').write_text(json.dumps(result,indent=2));(dest/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset))
    print('complete charts, per-dataset results and CSV saved')
if __name__=='__main__':main()
