import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import cv2
import os
import copy
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data.mixup import Mixup
# from timm.models import create_model

from PIL import Image

from src.utils import write_txt
from dataset.datasets import FasDataset
from src.utils import create_model
from torch.optim import lr_scheduler

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

class Model():

    def __init__(self, name='convnext_tiny', checkpoint_model='', model_prefix='', size=224):
        self.model = create_model(name)
        self.tfms = transforms.Compose([transforms.Resize(size), 
                                        # transforms.CenterCrop(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

        self.model.load_state_dict(torch.load(checkpoint_model)['model'])
        self.model.to('cuda:0')
        self.model.eval()

    def preprocess(self, x):
        inputs = []
        for xi in x:
            xi = cv2.imwrite("test.jpg", xi)
            img = self.tfms(Image.open('test.jpg')).cuda()
            # xi = self.tfms(Image.fromarray(xi[:,:,::-1]))
            inputs.append(img)
        inputs = torch.stack(inputs, dim=0)
        return inputs

    def predict(self, x):
        x = self.preprocess(x)
        with torch.no_grad():
            output = self.model(x)
            output = output.softmax(1).to('cpu').numpy()
        score = np.mean(output, axis=0)
        return score

def train(args, lenFolder):
    Dataset = FasDataset(args)
    train_size = int(0.8 * len(Dataset))
    val_size = len(Dataset) - train_size
    trainDataset, valDataset = torch.utils.data.random_split(Dataset, [train_size, val_size])
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, \
                             shuffle= True, num_workers= args.num_workers)
    valLoader = DataLoader(valDataset, batch_size=args.batch_size, \
                           shuffle= True, num_workers= args.num_workers)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    
    #----------------------------------
    model = torchvision.models.alexnet(pretrained = False)
    if args.activation == 'linear' :
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.num_classes)
        criterion = torch.nn.CrossEntropyLoss()
    else :
        model.classifier[-1] = nn.Sequential(
                                nn.Linear(model.classifier[-1].in_features, 1),
                                nn.Sigmoid()
                                )
        criterion = nn.BCELoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9)
    lr_schedule_values = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # model = create_model(args)
    model = model.to(device)
    # print(model)

    # model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    # # num_training_steps_per_epoch = len(trainDataset) // total_batch_size
    # num_training_steps_per_epoch = 1000 // total_batch_size
    # if args.layer_decay < 1.0 or args.layer_decay > 1.0:
    #     num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
    #     assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
    #          "Layer Decay impl only supports convnext_small/base/large/xlarge"
    #     assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    # else:
    #     assigner = None

    # if assigner is not None:
    #     print("Assigned values = %s" % str(assigner.values))
    
    # mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup is activated!")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.nb_classes)
    # if args.use_polyloss:
    #     criterion = Poly1CrossEntropyLoss(num_classes=args.nb_classes, reduction='mean')
    # elif mixup_fn is not None:
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    
    
    # optimizer = create_optimizer(
    #     args, model_without_ddp, skip_list=None,
    #     get_num_layer=assigner.get_layer_id if assigner is not None else None, 
    #     get_layer_scale=assigner.get_scale if assigner is not None else None)
    
    
    # lr_schedule_values = utils.cosine_scheduler(
    #     args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
    #     warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    # )
    

    best_acc = 0.0
    best_epoch = None
    for epoch in range(args.start_epoch, args.epochs +1):
        print(f'Epoch {epoch}/{args.epochs}')
        print('-' * 15)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs in tqdm(dataLoader[phase]):
                input = inputs['img_add'].to(device)
                if args.activation == 'linear' :
                    labels = inputs['label'].to(device)
                else :    
                    labels = inputs['label'].to(device).unsqueeze(-1).to(torch.float32)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input)
                    if args.activation == 'linear' :
                        _, preds = torch.max(outputs, 1)
                    else :
                         preds = (outputs > 0.5).float()
                    
                    # print(outputs)
                    # print(preds)
                    # print(labels)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_schedule_values.step()
            # print(running_loss)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if epoch % args.num_save_ckp ==0 or epoch == args.epochs or epoch == 1:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.checkpoint_dir, args.name_model, \
                            lenFolder,\
                                str(epoch)+'.pth'))
            
    model.load_state_dict(best_model_wts)
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.checkpoint_dir, args.name_model, \
                            lenFolder, ("best_epoch"+".pth"))) 
        
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')
    # Model parameters
    parser.add_argument('--name_model', type=str, default='alexnet')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--checkpoint_dir', type= str, default='checkpoints')
    parser.add_argument('--pretrained', type=str2bool, default= False)
    parser.add_argument('--num_save_ckp', type= int, default= 5)
    parser.add_argument('--threshold', type= float, default= 0.5)


    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--activation', type= str, default= 'linear', choices=['linear', 'sigmoid'])


    # Dataset parameters
    parser.add_argument('--path_data', default='data/train/0', type=str,
                        help='dataset path')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--load_height', type=int, default=224)
    parser.add_argument('--load_width', type=int, default=128)

    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_num', default=2, type=int)

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = get_args_parser()
    print('\n'.join(map(str,(str(args).split('(')[1].split(',')))))
    if not os.path.exists(args.checkpoint_dir) :
        os.mkdir(args.checkpoint_dir)

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.name_model)) :
        os.mkdir(os.path.join(args.checkpoint_dir, args.name_model))

    lenFolder = str(len(os.listdir(os.path.join(args.checkpoint_dir, args.name_model)))).zfill(3)
    if args.save_ckpt :
        os.mkdir(os.path.join(args.checkpoint_dir, args.name_model, \
                          lenFolder))
        arg_save = '\n'.join(map(str,(str(args).split('(')[1].split(','))))
        write_txt(arg_save, os.path.join(args.checkpoint_dir, args.name_model,\
            lenFolder, 'args.txt'))
    
    start_time = time.time()
    train(args= args, lenFolder = lenFolder)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


