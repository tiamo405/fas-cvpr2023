# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import cv2
import numpy as np
from tqdm import tqdm

from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.data import create_transform
# from dataset.FaceAligner import FaceAligner
from torch.utils import data
from torchvision import transforms
from dataset.utils import align_face, read_txt
from PIL import Image


class FasDatasetTest(data.Dataset):
    def __init__(self, args) -> None:
        super(FasDatasetTest, self).__init__()

        self.path_data = args.path_data
        self.load_height = args.load_height
        self.load_width = args.load_width
        self.parse = args.parse
        self.input = args.img_input
        self.rate = args.rate
        self.path_txt = args.path_txt
        self.resize = args.resize
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
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                
            ])
        self.rate = args.rate
        self.nb_classes = args.nb_classes
        path_image_s = []
        fnames = []

        with open(args.path_txt, 'r') as f :
            for line in f :
                fnames.append(line.split()[0])
        for fname in tqdm(fnames) :
            path_image = os.path.join(self.path_data, fname)
            path_image_s.append(path_image)
        self.path_image_s = path_image_s
    
    def __getitem__(self, index) :
        path_image  = self.path_image_s[index]

        img_full = cv2.imread(path_image) # anh full

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

        img_full_ycbcr = cv2.cvtColor(img_full, cv2.COLOR_RGB2YCrCb)[:,:,0]
        img_face_ycbcr = cv2.cvtColor(img_face, cv2.COLOR_RGB2YCrCb)[:,:,0].reshape((self.load_height, self.load_width, 1))
        img_align_ycbcr = cv2.cvtColor(img_align, cv2.COLOR_RGB2YCrCb)[:,:,0].reshape((self.load_height, self.load_width, 1))
        img_face_ycbcr = np.concatenate((img_face_ycbcr, img_face_ycbcr, img_face_ycbcr), axis= 2)
        img_align_ycbcr = np.concatenate((img_align_ycbcr, img_align_ycbcr, img_align_ycbcr), axis= 2)

        # transform
        img_full                         = self.transform(img_full)
        img_align                  = self.transform(img_align)
        img_face                        = self.transform(img_face)
        img_full_add_img_align               = self.transform(img_full_add_img_align)
        img_face_add_img_align               = self.transform(img_face_add_img_align)
        img_face_ycbcr                       = self.transform(img_face_ycbcr)
        img_align_ycbcr                       = self.transform(img_align_ycbcr)

        result = {
            'path_image' : path_image,
            'img_align' : img_align,
            'img_full' : img_full,
            'img_face' : img_face,
            'img_full_add_img_align' : img_full_add_img_align,
            'img_face_add_img_align' : img_face_add_img_align,
            'img_face_ycbcr' :img_face_ycbcr,
            'img_align_ycbcr' : img_align_ycbcr
        }
        return result
    
    def __len__(self):
        return len(self.path_image_s)