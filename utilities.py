import os
import shutil
import torch

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, is_best, title, filename='checkpoint.pth.tar'):
    filepath = title + '-' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, title + '-best.pth.tar')
