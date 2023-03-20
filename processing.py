import cv2
import pandas as pd
import torch
import argparse
import numpy as np
import os
import torchvision
import pandas as pd
import random
from torchvision import transforms
from PIL import Image
from torch import nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dataset.utils import align_face, read_txt

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
def trans(img, opt) :
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    img = cv2.resize(img, (opt.load_width,opt.load_height))
    img = transform(img)
    return img

class Model():

    def __init__(self, args = None):
        self.model = torchvision.models.alexnet(pretrained = False)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 2)
        self.tfms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        self.checkpoint_model = "checkpoints/alexnet/best_epoch.pth"
        self.model.load_state_dict(torch.load(self.checkpoint_model)['model_state_dict'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, path_image):
        img = cv2.imread(path_image)
        (left, top), (right, bottom), dst = read_txt(path_image.replace('.jpg', '.txt'))
        img_aligin = align_face(img, dst)
        img_pil_aligin = self.tfms(Image.fromarray(img_aligin))
        img = trans(img = img, opt= args).to(self.device).unsqueeze(0)
        return img
    def predict(self, path_image):
        img = self.preprocess(path_image)
        with torch.no_grad():
            output = self.model(img)
            output = output.softmax(1).to('cpu').numpy()
        score = np.mean(output, axis=0)
        return score
    

def pred(args, folder_save) :
    path_save = os.path.join(folder_save, 'filter_image.csv')
    model = Model(args = args)
    scores = []
    path = []
    df = pd.read_csv("data/train/all_image.csv")
    for pt in df['path_all_image'] :
        score = model.predict(pt)
        label = 'spoof' if np.argmax(score) ==0 else 'living'
        if label in pt and score[np.argmax(score)] >= args.threshold :
            print(f'path image: {pt}, score : {score}, label: {label}')
            scores.append(score)
            path.append(pt)
    # save_txt(path_save= path_save, fnames= path, scores= scores, remove= False)
    df = pd.DataFrame({
        'path' : path,
        'score' : scores
    })
    df.to_csv(path_save, index= False)
def themanh() :
    import zipfile
    path_train = []
    path = []
    with open('data/txt/pathImageTrain.txt', 'r') as f:
        for line in f :
            # print(line.split()[0].split('-'))
            path_train.append(line.split()[0])
    df = pd.read_csv("data/csv/all_image.csv")
    
    for pt in df['path_all_image'] :
        if 'living' in pt and pt not in path_train :
            path.append(pt)
    save = random.sample(path, 20000)
    zip_filename = os.path.join("data/train", '20000living.zip')
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pt in save : 
            print(pt)
            tmp = pt.split("/")
            arcname = tmp[-3]+'-'+tmp[-2]+'-'+tmp[-1]
            zipf.write(pt, arcname)
            zipf.write(pt.replace('.jpg', '.txt'), arcname.replace('.jpg', '.txt'))
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_csv', type= str2bool, default=True)
    parser.add_argument('--path_data', type= str, default= 'data')
    parser.add_argument('--path_save', type= str, default= 'results')
    parser.add_argument('--parse', type= str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--load_checkpoint', type= str2bool, default= False)
    parser.add_argument('--name_model', type=str, default= 'convnext_tiny')
    parser.add_argument('--num_train', type= str, default='1')
    parser.add_argument('--num_ckp', type=str, default='best_epoch')
    parser.add_argument('--threshold', type= float, default= 0.7)
    parser.add_argument('--load_height', type=int, default=480)
    parser.add_argument('--load_width', type=int, default=360)
    args = parser.parse_args()
    return args

if __name__ == "__main__" :
    args = get_args_parser()
    # pred(folder_save = "data/train", args= args)
    themanh()