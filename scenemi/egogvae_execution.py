"""Launch manually requested EgoGVAE actions through the existing resource guard."""
import argparse
from pathlib import Path
import sys

import yaml

from scenemi.config import ROOT


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','preflight','evaluate','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/egogvae_finetune.yaml')
    parser.add_argument('--device',choices=('cpu','cuda'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--per-dataset',type=int,default=8)
    args=parser.parse_args(argv)
    if args.resume and args.action!='train':parser.error('--resume is for training only')
    if args.checkpoint is not None and args.action!='evaluate':parser.error('--checkpoint is for evaluation only')
    config=args.config.resolve()
    c=yaml.safe_load(config.read_text());r=c['runtime']
    if ROOT/c['output']==ROOT/'runs/egogvae_finetune':
        from scenemi.egogvae_sharing import main as execute
        forwarded=['run',args.action,'--config',str(config)]
        forwarded+=['--device',args.device or ('cpu' if args.action in ('audit','preflight') else 'cuda'),
                    '--per-dataset',str(args.per_dataset)]
        if args.resume:forwarded.append('--resume')
        if args.checkpoint is not None:forwarded+=['--checkpoint',str(args.checkpoint.resolve())]
        return execute(forwarded)
    if not 0<r['memory_gib']<=16 or not 1<=r['cpu_count']<=4:
        parser.error('Require RAM <=16GiB and <=4 CPU cores')
    device=args.device or ('cuda' if args.action in ('train','evaluate') else 'cpu')
    child=[sys.executable,'-m','scenemi','egogvae',args.action,'--config',str(config),
           '--device',device,'--per-dataset',str(args.per_dataset)]
    if args.resume:child.append('--resume')
    if args.checkpoint is not None:child+=['--checkpoint',str(args.checkpoint.resolve())]
    # Reuse user-manager discovery, admission, cgroups and CPU affinity unchanged.
    from scenemi.execution import resource_main
    return resource_main(['run','--device',device,'--memory-gib',str(r['memory_gib']),
                          '--cpu-count',str(r['cpu_count']),'--',*child])
