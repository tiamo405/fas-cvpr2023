import argparse
import datetime
import time
import torch
import os
import copy

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler

from utils.utils import write_txt
from utils import utils_config_train, utils_model, utils_loss, utils_save_cfg
from dataset.datasets import FasDataset
from logs.logs import setup_logger
from engine import train_one_epoch
logger = setup_logger("train.log")
dir_name = os.path.dirname(os.path.realpath(__file__))



def train(cfg):
    BATCH_SIZE = cfg['BATCH_SIZE']
    EPOCHS = cfg['EPOCHS']
    TRAIN_ON = cfg['TRAIN_ON']
    DATA_ROOT = cfg['DATA_ROOT']
    CHECKPOINT_DIR = cfg['CHECKPOINT_DIR']
    #model
    NAME_MODEL =cfg['NAME_MODEL']
    NUM_CLASSES = cfg['NUM_CLASSES']
    LR = cfg['LR']
    NUM_WORKERS = cfg['NUM_WORKERS']
    NAME_LOSS = cfg['NAME_LOSS']
    DEVICE = cfg['DEVICE']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    PIN_MEMORY = cfg['PIN_MEMORY']
    #ckpt
    NUM_SAVE_CKPT = cfg['NUM_SAVE_CKPT']
    SAVE_CKPT = cfg['SAVE_CKPT']
    #data
    RESIZE = cfg['RESIZE']
    LOAD_WIDTH = cfg['LOAD_WIDTH']
    LOAD_HEIGHT = cfg['LOAD_HEIGHT']
    IMG_INPUT = cfg['IMG_INPUT']

    num_save_file = str(len(os.listdir(os.path.join(CHECKPOINT_DIR, NAME_MODEL)))).zfill(4)
    logger.info("train ")
    if SAVE_CKPT :
        utils_save_cfg.save_cfg(cfg= cfg, checkpoint_dir= CHECKPOINT_DIR, name_model= NAME_MODEL, num_save_file= num_save_file)

    Dataset = FasDataset(path_data= DATA_ROOT, load_width= LOAD_WIDTH,\
                          load_height= LOAD_HEIGHT, resize= RESIZE, num_classes= NUM_CLASSES)
    train_size = int(0.8 * len(Dataset))
    val_size = len(Dataset) - train_size
    trainDataset, valDataset = torch.utils.data.random_split(Dataset, [train_size, val_size])
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, \
                             shuffle= True, num_workers= NUM_WORKERS, pin_memory= PIN_MEMORY)
    valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, \
                           shuffle= True, num_workers= NUM_WORKERS, pin_memory= PIN_MEMORY)
    dataset_sizes = {
        'train' : len(trainDataset),
        'val' : len(valDataset),
    }
    dataLoader = { 
        'train' : trainLoader,  
        'val': valLoader
    }
    # print(Dataset.__getitem__(100))
    # ---------------------------------------------
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device :", DEVICE)
    
    #----------------------------------
    model = utils_model.create_model(name_model= NAME_MODEL, num_classes= NUM_CLASSES)
    model.to(device= DEVICE)

    criterion = utils_loss.create_loss(name_loss= NAME_LOSS, num_classes= NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR,  weight_decay=WEIGHT_DECAY)
    lr_schedule_values = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    best_epoch = None
    for epoch in range(1, EPOCHS +1):
        result = train_one_epoch(model= model, criterion= criterion, optimizer= optimizer, 
                                 lr_schedule_values= lr_schedule_values,
                                 device= DEVICE, epoch= epoch, epochs= EPOCHS, dataset_sizes= dataset_sizes,
                                 dataLoader = dataLoader,
                                 img_input= IMG_INPUT, num_save_ckpt= NUM_SAVE_CKPT,
                                 save_ckpt= SAVE_CKPT, name_model= NAME_MODEL, train_on= TRAIN_ON, 
                                 checkpoint_dir= CHECKPOINT_DIR, num_save_file= num_save_file)
        if result['val_acc'] > best_acc :
            best_acc = result['val_acc']
            best_model_wts = result['best_model_wts']
            best_epoch = epoch
    model.load_state_dict(best_model_wts)
    if TRAIN_ON == 'ssh' :
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, NAME_MODEL, \
                        num_save_file, ("best_epoch"+".pth"))) 
    else : 
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
        }, ("best_epoch.pth"))
        
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_config', type= int, default= 1)
    
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = get_args_parser()
    cfg = utils_config_train.config[args.num_config]
    
    start_time = time.time()
    train(cfg= cfg)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


