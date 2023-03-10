# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
# import pandas as pd
import cv2
import numpy as np
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.data import create_transform
# from dataset.FaceAligner import FaceAligner
from torch.utils import data
from torchvision import transforms
from dataset.utils import align_face, read_txt
from PIL import Image

# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     print("Transform = ")
#     if isinstance(transform, tuple):
#         for trans in transform:
#             print(" - - - - - - - - - - ")
#             for t in trans.transforms:
#                 print(t)
#     else:
#         for t in transform.transforms:
#             print(t)
#     print("---------------------------")

#     if args.data_set == 'CIFAR':
#         dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
#         nb_classes = 100
#     elif args.data_set == 'IMNET':
#         print("reading from datapath", args.data_path)
#         root = os.path.join(args.data_path, 'train' if is_train else 'val')
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif args.data_set == "image_folder":
#         root = args.data_path if is_train else args.eval_data_path
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = args.nb_classes
#         assert len(dataset.class_to_idx) == nb_classes
#     else:
#         raise NotImplementedError()
#     # print("Number of the class = %d" % nb_classes)

#     return dataset, nb_classes


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
#     mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

#     if is_train:
#         # this should always dispatch to transforms_imagenet_train

#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             hflip=args.hflip,
#             vflip=args.vflip,
#             mean=mean,
#             std=std,
#         )

#         if not resize_im:
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform


#     t = []
#     if resize_im:
#         # warping (no cropping) when evaluated at 384 or larger
#         if args.input_size >= 384:  
#             t.append(
#             transforms.Resize((args.input_size, args.input_size), 
#                             interpolation=transforms.InterpolationMode.BICUBIC), 
#         )
#             print(f"Warping {args.input_size} size input images...")
#         else:
#             if args.crop_pct is None:
#                 args.crop_pct = 224 / 256
#             size = int(args.input_size / args.crop_pct)
#             t.append(
#                 # to maintain same ratio w.r.t. 224 images
#                 transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
#             )
#             t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)

class FasDataset(data.Dataset):
    def __init__(self, args) -> None:
        super(FasDataset, self).__init__()
        self.path_data = args.path_data
        
        self.load_height = args.load_height
        self.load_width = args.load_width
        self.transform = transforms.Compose([
            #ColorJitter() thực hiện việc thay đổi độ sáng, 
            # độ tương phản, độ bão hòa màu và màu sắc của hình ảnh.
            
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Resize((self.load_height, self.load_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            
        ])
        self.rate = args.rate
        self.nb_classes = args.nb_classes
        path_image_s = []
        labels = []
        for folder in os.listdir(self.path_data) :
            for pt in os.listdir(os.path.join(self.path_data, folder)) :
                if '.txt' not in pt :
                    path_image_s.append(os.path.join(self.path_data, folder, pt))
                    if self.nb_classes == 2 :
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
        (left, top), (right, bottom), dst = read_txt(path_image.replace('.jpg', '.txt'))
        left, top, right, bottom = int(left/ self.rate), int( top / self.rate), int(right * self.rate), int(bottom * self.rate)
        # print(img.shape)
        img_aligin      = align_face(img_full, dst) # ảnh face
        img_rate = img_full[top: bottom, left: right, :]

        img_full                         =cv2.resize(img_full, (self.load_width, self.load_height))
        img_rate                        =cv2.resize(img_rate, (self.load_width, self.load_height))
        img_aligin                      = cv2.resize(img_aligin, (self.load_width, self.load_height))
        
        img_add_img_full_aligin         = np.concatenate((img_full, img_aligin), axis= 1)
        img_add_img_rate_aligin         = np.concatenate((img_rate, img_aligin), axis= 1)
        # đưa về cùng kích cỡ width*2, height
        img_full = img_full             =cv2.resize(img_full, (self.load_width*2, self.load_height))
        img_rate                        =cv2.resize(img_rate, (self.load_width*2, self.load_height))
        img_aligin                      = cv2.resize(img_aligin, (self.load_width*2, self.load_height))
        # cv2 to Image PIL
        img_full                         = Image.fromarray(img_full)
        img_aligin                      = Image.fromarray(img_aligin)
        img_rate                        = Image.fromarray(img_rate)
        img_add_img_full_aligin          = Image.fromarray(img_add_img_full_aligin)
        img_add_img_rate_aligin           = Image.fromarray(img_add_img_rate_aligin)

        # transform
        img_full                         = self.transform(img_full)
        img_aligin                  = self.transform(img_aligin)
        img_rate                        = self.transform(img_rate)
        img_add_img_full_aligin               = self.transform(img_add_img_full_aligin)
        img_add_img_rate_aligin               = self.transform(img_add_img_rate_aligin)


        result = {
            'path_image' : path_image,
            'label' : label,
            'img_pil_aligin' : img_aligin,
            'img_full' : img_full,
            'img_add_img_full_aligin' : img_add_img_full_aligin,
            'img_add_img_rate_aligin' : img_add_img_rate_aligin
        }
        return result
    
    def __len__(self):
        return len(self.path_image_s)