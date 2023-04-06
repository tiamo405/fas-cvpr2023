import cv2
import pandas as pd
import torch
import argparse
import numpy as np
import os
import torchvision
import datetime
import shutil
import time

from torch import nn
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dataset.utils import align_face, read_txt
from utils.utils import write_txt, save_zip, str2bool
from tqdm import tqdm
from dataset.dataset_test import FasDatasetTest
from torch.utils.data import DataLoader
from utils import  utils_config_predict, utils_model

# class SplitModel(nn.Module):
#     def __init__(self):
#         super(SplitModel, self).__init__()
#         self.resnet = resnet101()
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, 2)

#     def forward(self, x):
#         x = self.resnet(x)

#         return x
# class ResNetModified(nn.Module):
#     def __init__(self):
#         super(ResNetModified, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained= False)
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, 2)

#     def forward(self, x):
#         x = self.resnet(x)
#         return x

# class AlexnetModified(nn.Module) :
#     def __init__(self, args):
#         super(AlexnetModified, self).__init__()
#         self.model = torchvision.models.alexnet(pretrained = args.pretrained)
#         if args.activation == 'linear' :
#             self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, args.nb_classes)
#         else :
#             self.model.classifier[-1] = nn.Sequential(
#                                     nn.Linear(self.model.classifier[-1].in_features, 1),
#                                     nn.Sigmoid()
#                                     )
#     def forward(self, x):
#         x = self.model(x)
#         return x

# class Resnet50Edit(nn.Module) :
#     def __init__(self):
#         super(ResNetModified, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=False)
#         for param in self.model.parameters():
#             param.requires_grad = False
#         n_inputs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Sequential(
#                     nn.Linear(n_inputs, 256),
#                     nn.ReLU(),
#                     nn.Dropout(0.5),
#                     nn.Linear(256, 2),
#                     nn.LogSoftmax(dim=1)
#                 )
#     def forward(self, x):
#         x = self.resnet(x)
#         return x
    
class Model():

    def __init__(self, name_model, nb_classes, load_height, load_width, resize , img_input,
                 checkpoint_dir, num_train, num_ckpt):
        
        self.model = utils_model.create_model(name_model= name_model, num_classes= nb_classes)
        self.nb_classes = nb_classes
        self.resize = resize
        self.load_height = load_height
        self.load_width = load_width

        if self.resize == True :
            self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((self.load_height, self.load_width)),
                                transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else : 
            self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        self.input =img_input
        
        self.checkpoint_model = os.path.join(checkpoint_dir, name_model, num_train, num_ckpt+'.pth')
 
        self.model.load_state_dict(torch.load(self.checkpoint_model)['model_state_dict'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        

    def preprocess(self, path_image):
        img_full = cv2.imread(path_image)
        try :
            (left, top), (right, bottom), dst = read_txt(path_image.replace('.jpg', '.txt'))
        except :
            (left, top), (right, bottom), dst = read_txt(path_image.replace('.png', '.txt'))
        
        
        img_face = img_full[top: bottom, left: right, :]
        img_align = align_face(img_full, dst)

        img_full  =cv2.resize(img_full, (self.load_width, self.load_height))
        img_align = cv2.resize(img_align, (self.load_width, self.load_height))

        img_full_add_img_align         = np.concatenate((img_full, img_align), axis= 1)
        img_face_add_img_align         = np.concatenate((img_face, img_align), axis= 1)
        

        img_full                         = self.transform(img_full)
        img_align                      = self.transform(img_align)
        img_face                        = self.transform(img_face)
        img_full_add_img_align           = self.transform(img_full_add_img_align )
        img_face_add_img_align          = self.transform(img_face_add_img_align)

        if self.input == 'img_full' :
            return img_full.to(self.device).unsqueeze(0)
        elif self.input == 'img_full_add_img_align' :
            return img_full_add_img_align .to(self.device).unsqueeze(0)
        else:
            return img_face_add_img_align.to(self.device).unsqueeze(0)
    def predict(self, path_image):
        input = self.preprocess(path_image)
        # self.model.eval()
        with torch.no_grad():
            output = self.model(input) 
            if self.activation == 'sigmoid':
                output = output.to('cpu').numpy()
            else :
                output = output.softmax(1).to('cpu').numpy()
            
        score = np.mean(output, axis=0)
        return score


def pred_old(args) :
    path_save_txt = os.path.join(folder_save, 'submit.txt')
    if PHASE == 'dev' : 
        args.path_txt = "/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/Dev.txt"
        args.path_data = '/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/data'
        
    else : 
        args.path_txt = "/mnt/sda1/datasets/FAS-CVPR2023/test/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Test_V2-20230223/Test.txt"
        args.path_data = '/mnt/sda1/datasets/FAS-CVPR2023/test/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Test_V2-20230223/data'
        shutil.copy(os.path.join(args.path_save, 'dev', args.combine, 'submit.txt'), folder_save)
    model = Model(args = args)
    print(model.model)
    fnames = []
    with open(args.path_txt, 'r') as f :
        for line in f :
            fnames.append(line.split()[0])
    scores = []
    path = []
    for fname in tqdm(fnames) :
        path_image = os.path.join(args.path_data, fname)
        score = model.predict(path_image= path_image)

        if args.save_txt :
            if args.activation == 'linear':
                if args.nb_classes == 2 :
                    write_txt(noidung= PHASE + '/'+ fname + ' ' + "{}".format(np.argmax(score)), 
                      path= path_save_txt)

                else :

                    write_txt(noidung= PHASE + '/'+ fname + ' ' + "{}".format(score[1]), 
                      path= path_save_txt)

            else :
                write_txt(noidung= PHASE + '/'+ fname + ' ' + "{}".format(score[0]* args.threshold), 
                      path= path_save_txt)
    if args.save_txt :
        save_zip(folder_save= folder_save)
        print('save success {}'.format(folder_save))
#----------------------------

def pred_new(cfg) :

    SAVE_TXT = cfg['SAVE_TXT']
    PHASE = cfg['PHASE']
    DEVICE = cfg['DEVICE']
    PATH_DATA = cfg['PATH_DATA']
    PATH_SAVE = cfg['PATH_SAVE']
    PATH_TXT = cfg['PATH_TXT']
    CHECKPOINT_DIR = cfg['CHECKPOINT_DIR']
    NAME_MODEL = cfg['NAME_MODEL']
    NUM_CLASSES = cfg['NUM_CLASSES']
    THRESHOLD = cfg['THRESHOLD']
    NUM_TRAIN = cfg['NUM_TRAIN']
    NUM_CKPT = cfg['NUM_CKPT']

    RESIZE = cfg['RESIZE']
    LOAD_WIDTH = cfg['LOAD_WIDTH']
    LOAD_HEIGHT =cfg['LOAD_HEIGHT']
    IMG_INPUT = cfg['IMG_INPUT']
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_WORKERS = cfg['NUM_WORKERS']
    COMBINE = cfg['COMBINE']
    SHUFFLE = cfg['SHUFFLE']

    folder_save = os.path.join(PATH_SAVE, PHASE, 
                               str(len(os.listdir(os.path.join(PATH_SAVE, PHASE)))).zfill(4))
    os.makedirs(folder_save, exist_ok= True)
    path_save_txt = os.path.join(folder_save, 'submit.txt')
    if PHASE == 'dev' : 
        # args.path_txt = "data/txt/dev.txt"
        PATH_TXT = "/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/Dev.txt"
        PATH_DATA = '/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/data'
        
    else : 
        # args.path_txt = "data/txt/Test.txt"
        PATH_TXT = "/mnt/sda1/datasets/FAS-CVPR2023/test/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Test_V2-20230223/Test.txt"
        PATH_DATA = '/mnt/sda1/datasets/FAS-CVPR2023/test/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Test_V2-20230223/data'
        if COMBINE != '000' :
            shutil.copy(os.path.join(PATH_SAVE, 'dev', COMBINE, 'submit.txt'), folder_save)


    model = Model(name_model= NAME_MODEL, nb_classes= NUM_CLASSES, load_height= LOAD_HEIGHT,
                  load_width= LOAD_WIDTH, resize= RESIZE, img_input= IMG_INPUT,
                  checkpoint_dir= CHECKPOINT_DIR, num_train= NUM_TRAIN, num_ckpt= NUM_CKPT).model
    print(model)

    testDataset = FasDatasetTest(path_data= PATH_DATA, load_height= LOAD_HEIGHT, load_width= LOAD_WIDTH,
                                 path_txt= PATH_TXT, resize= RESIZE, nb_classes= NUM_CLASSES)
    testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, \
                            num_workers= NUM_WORKERS, shuffle= SHUFFLE)


    fnames = []
    with open(PATH_TXT, 'r') as f :
        for line in f :
            fnames.append(line.split()[0])

    scores = []

    for inputs in tqdm(testLoader):

        input  = inputs[IMG_INPUT].to(DEVICE)
        with torch.no_grad() :
            output = model(input)
            scores = output.softmax(1).to('cpu').numpy()
        for i in range(len(input)):
            fname = inputs['path_image'][i].split('/')[-1]
            score = scores[i]
            if SAVE_TXT  :
                if NUM_CLASSES == 2 :
                    write_txt(noidung= PHASE + '/'+ fname + ' ' + "{}".format(score[1]), 
                    path= path_save_txt)
                else :
                    if score[1] >= THRESHOLD : 
                        write_txt(noidung= PHASE + '/'+ fname + ' ' + "{}".format(score[1]), 
                        path= path_save_txt)
                    else :
                        write_txt(noidung= PHASE + '/'+ fname + ' ' + "{}".format(abs(score[1] - args.threshold)), 
                    path= path_save_txt)


    if SAVE_TXT :
        save_zip(folder_save= folder_save)
        print('save success {}'.format(folder_save))

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_config', type= int, default= 1)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__" :
    args = get_args_parser()
    cfg = utils_config_predict.config[args.num_config]
    PHASE = cfg['PHASE']
    COMBINE = cfg['COMBINE'] 
    start_time = time.time()
    # pred_old(folder_save = folder_save, args= args)
    if COMBINE == '000' and PHASE == 'test':
        PHASE = 'dev'
        pred_new(cfg=cfg)
        PHASE = 'test'

    pred_new(cfg=cfg)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Predict time {}'.format(total_time_str))
    
    