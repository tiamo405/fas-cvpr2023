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
from src.utils import write_txt, save_zip, str2bool
from tqdm import tqdm

class Model():

    def __init__(self, args = None):
        self.model = torchvision.models.alexnet(pretrained = False)
        self.nb_classes = args.nb_classes
        self.activation = args.activation
        if self.activation == 'linear' :
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.nb_classes)
        else :
            self.model.classifier[-1] = nn.Sequential(
                                        nn.Linear(self.model.classifier[-1].in_features, 1),
                                        nn.Sigmoid()
                                        )
        self.load_height = args.load_height
        self.load_width = args.load_width
        self.transform = transforms.Compose([
                                        transforms.Resize((self.load_height, self.load_width)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        self.input = args.img_input
        self.rate = args.rate
        
        self.checkpoint_model = os.path.join(args.checkpoint_dir, args.name_model, args.num_train, args.num_ckpt+'.pth')
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
        left, top, right, bottom = int(left/ self.rate), int( top / self.rate), int(right * self.rate), int(bottom * self.rate)
        
        img_rate = img_full[top: bottom, left: right, :]
        img_aligin = align_face(img_full, dst)
        img_rate   =cv2.resize(img_rate, (self.load_width, self.load_height))
        img_full  =cv2.resize(img_full, (self.load_width, self.load_height))
        img_aligin = cv2.resize(img_aligin, (self.load_width, self.load_height))

        img_full_add_img_aligin         = np.concatenate((img_full, img_aligin), axis= 1)
        img_rate_add_img_aligin         = np.concatenate((img_rate, img_aligin), axis= 1)
        
        img_full                         = Image.fromarray(img_full)
        img_aligin                      = Image.fromarray(img_aligin)
        img_rate                        = Image.fromarray(img_rate)
        img_full_add_img_aligin           = Image.fromarray(img_full_add_img_aligin )
        img_rate_add_img_aligin           = Image.fromarray(img_rate_add_img_aligin)

        img_full                         = self.transform(img_full)
        img_aligin                      = self.transform(img_aligin)
        img_rate                        = self.transform(img_rate)
        img_full_add_img_aligin           = self.transform(img_full_add_img_aligin )
        img_rate_add_img_aligin          = self.transform(img_rate_add_img_aligin)

        if self.input == 'img_full' :
            return img_full.to(self.device).unsqueeze(0)
        elif self.input == 'img_full_add_img_aligin' :
            return img_full_add_img_aligin .to(self.device).unsqueeze(0)
        else:
            return img_rate_add_img_aligin.to(self.device).unsqueeze(0)
    def predict(self, path_image):
        input = self.preprocess(path_image)
        with torch.no_grad():
            output = self.model(input) 
            if self.activation == 'sigmoid':
                output = output.to('cpu').numpy()
            else :
                output = output.softmax(1).to('cpu').numpy()
            
        score = np.mean(output, axis=0)
        return score

def pred(args, folder_save) :
    path_save_txt = os.path.join(folder_save, 'submit.txt')
    if args.parse == 'dev' : 
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
        # print(score)
        # scores.append(score[-1])
        # path.append(args.parse + '/'+ fname)
        # if args.activation == 'linear':
        #     print(f'path: {fname}, score: {score} ', end='')
        #     print('label: ', 'living' if np.argmax(score) == 1 else 'spoof')
        # else :
        #     print(f'path: {fname}, score: {score} ', end= '')
        #     print('label: ','living' if score[0] > args.threshold else 'spoof' )

        if args.save_txt :
            if args.activation == 'linear':
                if args.nb_classes == 2 :
                    # write_txt(noidung= args.parse + '/'+ fname + ' ' + "{}".format(np.argmax(score)), 
                    #   path= path_save_txt)
                    write_txt(noidung= args.parse + '/'+ fname + ' ' + "{}".format(score[1]* args.threshold), 
                      path= path_save_txt)
                    if score[1] >= args.threshold : 
                        write_txt(noidung= args.parse + '/'+ fname + ' ' + "{}".format(score[1]), 
                        path= path_save_txt)
                    else :
                        write_txt(noidung= args.parse + '/'+ fname + ' ' + "{}".format(abs(score[1] - args.threshold)), 
                      path= path_save_txt)
                else :
                    write_txt(noidung= args.parse + '/'+ fname + ' ' + "{}".format(score[1]), 
                      path= path_save_txt)
                # write_txt(noidung= args.parse + '/'+ fname + ' ' + "{:.10f}".format(abs(score[1]-score[0])), 
                #       path= path_save_txt)
            else :
                write_txt(noidung= args.parse + '/'+ fname + ' ' + "{}".format(score[0]* args.threshold), 
                      path= path_save_txt)
    if args.save_txt :
        save_zip(folder_save= folder_save)
        print('save success {}'.format(folder_save))

def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_txt', type= str2bool, default=True)
    #path, dir
    parser.add_argument('--path_data', type= str, default= '/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/data')
    parser.add_argument('--path_save', type= str, default= 'results')
    parser.add_argument('--path_txt', type= str, default="data/dev/dev.txt")
    parser.add_argument('--checkpoint_dir', type= str, default= 'checkpoints')

    #model
    parser.add_argument('--activation', type= str, default= 'linear', choices=['linear', 'sigmoid'])
    parser.add_argument('--nb_classes', type= int, default= 2)
    parser.add_argument('--load_checkpoint', type= str2bool, default= False)
    parser.add_argument('--name_model', type=str, default= 'alexnet')
    parser.add_argument('--num_train', type= str)
    parser.add_argument('--num_ckpt', type=str)
    parser.add_argument('--threshold', type= float, default= 0.998)

    #data
    parser.add_argument('--parse', type= str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--load_height', type=int, default=224)
    parser.add_argument('--load_width', type=int, default=128)
    parser.add_argument('--img_input', type=str, default='img_rate_add_img_aligin', \
                        choices=['img_full','img_full_add_img_aligin', 'img_rate_add_img_aligin'])
    parser.add_argument('--rate', type=float, default=1.2)
    parser.add_argument('--combine', type= str, default= '016')
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__" :
    args = get_args_parser()
    print('\n'.join(map(str,(str(args).split('(')[1].split(',')))))
    folder_save = ''
    if not os.path.exists(args.path_save) :
        os.makedirs(args.path_save)

    if not os.path.exists(os.path.join(args.path_save, args.parse)) :
        os.makedirs(os.path.join(args.path_save, args.parse))
    
    if args.save_txt == True :
        num_folder = str(len(os.listdir(os.path.join(args.path_save, args.parse)))).zfill(3)

        if not os.path.exists(os.path.join(args.path_save, 
                                        args.parse, 
                                        num_folder)) :
                os.makedirs(os.path.join(args.path_save, 
                                        args.parse, 
                                        num_folder))
        
        folder_save = os.path.join(args.path_save, args.parse, str(num_folder) )
        arg_save = '\n'.join(map(str,(str(args).split('(')[1].split(','))))
        write_txt(arg_save, os.path.join(folder_save, 'args.txt'))
    start_time = time.time()
    pred(folder_save = folder_save, args= args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Predict time {}'.format(total_time_str))
    
    