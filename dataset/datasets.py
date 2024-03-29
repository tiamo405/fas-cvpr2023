# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import cv2
import numpy as np
from torchvision import datasets, transforms

from torch.utils import data
from torchvision import transforms
from dataset.utils import align_face, read_txt, rgb2Y_in_Ycbcr



class FasDataset(data.Dataset):
    def __init__(self, path_data, load_width, load_height, resize, num_classes) -> None:
        super(FasDataset, self).__init__()
        self.path_data =path_data
        self.load_height = load_height
        self.load_width = load_width
        self.resize = resize
        self.num_classes = num_classes

        if self.resize == True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.Resize((self.load_height, self.load_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                
            ])
        else : 
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                
            ])
        
        path_image_s = []
        labels = []
        for folder in os.listdir(self.path_data) :
            for pt in os.listdir(os.path.join(self.path_data, folder)) :
                if '.txt' not in pt :
                    path_image_s.append(os.path.join(self.path_data, folder, pt))

                    if self.num_classes == 2 :
                        labels.append(0 if 'spoof'in pt else 1)
                    else :
                        if 'spoof' in pt and '3D' not in pt :
                            labels.append(0)
                        elif 'living' in pt :
                            labels.append(1)
                        else : labels.append(2)
        self.path_image_s = path_image_s
        self.labels = labels
    
    def __getitem__(self, index) :
        path_image  = self.path_image_s[index]
        label       = self.labels[index]
        # print(path_image)
        img_full = cv2.imread(path_image) # anh full
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # chuyen mau
        try :
            (left, top), (right, bottom), dst = read_txt(path_image.replace('.jpg', '.txt'))
        except : 
             (left, top), (right, bottom), dst = read_txt(path_image.replace('.png', '.txt'))
        # left, top, right, bottom = int(left/ self.rate), int( top / self.rate), int(right * self.rate), int(bottom * self.rate)
        # print(img.shape)
        img_align      = align_face(img_full, dst) # ảnh face
        img_face = img_full[top: bottom, left: right, :]

        img_full                         =cv2.resize(img_full, (self.load_width, self.load_height))
        img_face                        =cv2.resize(img_face, (self.load_width, self.load_height))
        img_align                      = cv2.resize(img_align, (self.load_width, self.load_height))
        
        img_full_add_img_align         = np.concatenate((img_full, img_align), axis= 1)
        img_face_add_img_align         = np.concatenate((img_face, img_align), axis= 1)
        

        img_full_ycbcr = rgb2Y_in_Ycbcr(img_full)
        img_face_ycbcr = rgb2Y_in_Ycbcr(img_face)
        img_align_ycbcr = rgb2Y_in_Ycbcr(img_align)
        img_full_add_img_align_ycbcr = rgb2Y_in_Ycbcr(img_full_add_img_align)
        img_face_add_img_align_ycbcr = rgb2Y_in_Ycbcr(img_face_add_img_align)


        # transform
        img_full                         = self.transform(img_full)
        img_align                  = self.transform(img_align)
        img_face                        = self.transform(img_face)
        img_full_add_img_align               = self.transform(img_full_add_img_align)
        img_face_add_img_align               = self.transform(img_face_add_img_align)
        img_face_ycbcr                       = self.transform(img_face_ycbcr)
        img_align_ycbcr                       = self.transform(img_align_ycbcr)
        img_face_add_img_align_dim6 =   torch.cat((img_face, img_align), dim=0)

        
        result = {
            'path_image' : path_image,
            'label' : label,
            'img_align' : img_align,
            'img_face' : img_face,
            'img_full' : img_full,
            'img_full_add_img_align' : img_full_add_img_align,
            'img_face_add_img_align' : img_face_add_img_align,
            'img_face_ycbcr' :img_face_ycbcr,
            'img_align_ycbcr' : img_align_ycbcr,
            'img_face_add_img_align_dim6' : img_face_add_img_align_dim6
        }
        return result
    
    def __len__(self):
        return len(self.path_image_s)