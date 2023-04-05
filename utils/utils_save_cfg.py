import os
from utils.utils import write_txt

def save_cfg(cfg, checkpoint_dir, name_model, num_save_file):
    save = '\n'.join(map(str,(str(cfg).split(','))))
    write_txt(noidung= save, path = os.path.join(checkpoint_dir, name_model, \
                        num_save_file,'cfg.txt'))