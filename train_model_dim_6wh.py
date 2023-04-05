import argparse
import datetime

import time
import torch
import torch.nn as nn

import os
import copy
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm


from losses import Poly1CrossEntropyLoss
from pytorch_metric_learning import losses as los
from utils.utils import write_txt, str2bool
from dataset.datasets import FasDataset
from losses import ArcFace
from torch.optim import lr_scheduler
from models.resnet import resnet101

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.resnet = resnet101()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet(x)

        return x



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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    print("device :", device)
    
    #----------------------------------
    model = SplitModel()

    

    model.to(device)
    print(model)

    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    if args.loss == 'Poly1CrossEntropyLoss':
        criterion = Poly1CrossEntropyLoss(num_classes=args.nb_classes, reduction='mean')
    if args.loss == 'ArcFace' :
        # criterion = ArcFace(s = 10, margin= 0.5)
        criterion = los.ArcFaceLoss(2, embedding_size = 2, margin=28.6, scale=64)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay=5e-4)

    # # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)
    lr_schedule_values = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    best_acc = 0.0
    best_epoch = None
    for epoch in range(1, args.epochs +1):
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
                input  = inputs[args.img_input].to(device)

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
                    if args.loss == 'BCEWithLogitsLoss':
                        loss = criterion(outputs, torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=2).to(torch.float32))
                    if args.loss == 'Poly1CrossEntropyLoss' :
                        loss = criterion(outputs, labels)
                    if args.loss == 'ArcFace' :
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
            if (epoch % args.num_save_ckpt ==0 or epoch == args.epochs or epoch == 1 ) and args.save_ckpt:
                if args.train_on == 'ssh' :
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, os.path.join(args.checkpoint_dir, args.name_model, \
                                lenFolder,\
                                    str(epoch)+'.pth'))
                else :
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, str(epoch) +'.pth')
            
    model.load_state_dict(best_model_wts)
    if args.train_on == 'ssh' :
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(args.checkpoint_dir, args.name_model, \
                                lenFolder, ("best_epoch"+".pth"))) 
    else : 
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
        }, ("best_epoch"+ args.activation +".pth"))
        
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--parse', type=str, default= 'train')
    parser.add_argument('--train_on', type=str, default='ssh', choices=['kaggle', 'ssh'])
    # path, dir
    parser.add_argument('--checkpoint_dir', type= str, default='checkpoints')
    parser.add_argument('--path_data', default='data/train', type=str,
                        help='dataset path')
    
    # Model parameters
    parser.add_argument('--name_model', type=str, default='SplitModel',\
                         choices=['SplitModel'])
    parser.add_argument('--nb_classes', default=2, type=int)
    
    parser.add_argument('--activation', type= str, default= 'linear',\
                         choices=['linear', 'sigmoid'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--loss', type=str, default= 'ArcFace',\
                         choices=['BCEWithLogitsLoss', 'Poly1CrossEntropyLoss', 'ArcFace'])
    
    #checkpoint
    parser.add_argument('--num_save_ckpt', type= int, default= 5)
    parser.add_argument('--save_ckpt', type=str2bool, default=False)
    

    # Dataset parameters

    parser.add_argument('--resize', type=str2bool, default= True)
    parser.add_argument('--load_height', type=int, default=224)
    parser.add_argument('--load_width', type=int, default=224)
    parser.add_argument('--rate', type=float, default=1.2)
    parser.add_argument('--img_input', type=str, default='img_face_add_img_align_dim6', \
        choices=['img_face_add_img_align_dim6'])

    
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = get_args_parser()
    print('\n'.join(map(str,(str(args).split('(')[1].split(',')))))
    lenFolder = ''
    if args.train_on == 'ssh' :
        os.makedirs(os.path.join(args.checkpoint_dir, args.name_model), exist_ok= True)

        lenFolder = str(len(os.listdir(os.path.join(args.checkpoint_dir, args.name_model)))).zfill(3)
        if args.save_ckpt :
            os.mkdir(os.path.join(args.checkpoint_dir, args.name_model, \
                            lenFolder))
            arg_save = '\n'.join(map(str,(str(args).split('(')[1].split(','))))
            write_txt(arg_save, os.path.join(args.checkpoint_dir, args.name_model,\
                lenFolder, 'args.txt'), remove= True)
    else :
        arg_save = '\n'.join(map(str,(str(args).split('(')[1].split(','))))
        write_txt(arg_save, 'args.txt', remove= True)
    
    start_time = time.time()
    train(args= args, lenFolder = lenFolder)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


