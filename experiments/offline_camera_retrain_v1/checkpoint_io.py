"""Durable, same-directory checkpoint replacement."""
import os
from pathlib import Path
import torch

def save_checkpoint(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value,stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(str(path.parent),os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
