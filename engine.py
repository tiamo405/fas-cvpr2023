# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import copy
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import tqdm
import os
from src import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, dataLoader,
                    device = None, epoch = None, lr_schedule_values=None,
                    dataset_sizes = None, args = None
                    ):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        for inputs in tqdm(dataLoader[phase]):
            img = inputs['img'].to(device)
            img_pil_aligin = inputs['img_pil_aligin'].to(device)
            labels = inputs['label'].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(img_pil_aligin, img)
                # print(outputs)
                _, preds = torch.max(outputs, 1)
                # print(preds)
                loss = criterion(outputs, labels)
                # print(loss)
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

        if phase == 'val' :
            val_acc = epoch_acc
            val_model_wts = copy.deepcopy(model.state_dict())
        if epoch % args.num_save_ckp ==0 or epoch == args.epochs:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'anpha': args.anpha,
            'beta': args.beta
        }, os.path.join(args.checkpoint_dir, args.name_model, \
                          str(len(os.listdir(os.path.join(args.checkpoint_dir, args.name_model)))),\
                            str(epoch)+'.pth'))

    # load best model weights
    model.load_state_dict(val_model_wts)
    return ({'model': model, 
             'val_acc' : val_acc, 
             'optimizer' : optimizer,
              'loss': loss})
