# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import logging

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, lr_schedule_values : torch.optim, 
                    dataLoader: DataLoader,
                    device : None, epoch : int, epochs : int,
                    dataset_sizes : dict, img_input : str,
                    num_save_ckpt : int, save_ckpt : bool, name_model : str, 
                    train_on : str, checkpoint_dir : str, num_save_file : str, logger : logging
                    ):
    print(f'Epoch {epoch}/{epochs}')
    print('-' * 20)
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        for inputs in tqdm(iter(dataLoader[phase])):
            input = inputs[img_input].to(device)
            labels = inputs['label'].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(input)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * input.size(0)
            
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            lr_schedule_values.step()
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        logger.info(f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if phase == 'val' :
            val_acc = epoch_acc
            val_model_wts = copy.deepcopy(model.state_dict())

    if (epoch % num_save_ckpt ==0  or epoch == epochs or epoch == 1) and save_ckpt :
        if train_on == 'ssh' :
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, os.path.join(checkpoint_dir, name_model, \
                        num_save_file, str(epoch)+'.pth'))
        else :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, str(epoch) +'.pth')

    return ({'model': model, 
             'val_model_wts' : val_model_wts,
             'val_acc' : val_acc, 
             'optimizer' : optimizer,
              'loss': epoch_loss})
