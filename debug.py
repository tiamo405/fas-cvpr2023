import cv2
import os
import json
import torch.nn as nn
import torch
import zipfile as zf    
import pandas as pd
import numpy as np
import zipfile
import random
from dataset.utils import read_txt, align_face
def checkReturn():
    a = 2
    b = 3
    return ({
        'a' :a,
        'b' :b
    })

def softmax():
    m = nn.Softmax(dim=1)
    input = torch.randn(3, 2)
    output = m(input)
    print(input)
    print(output)

def checkdict() :
    dic = {
            '2D-Display-Pad' : [0,600],
            '2D-Display-Phone' : [1,3000],
            '2D-Print-Album' :[2,3000],
            '2D-Print-Newspaper' :[3,3000],
            '2D-Print-Photo' :[4,3000],
            '2D-Print-Poster' :[5,3000],
            '3D-AdultDoll' :[6,165],
            '3D-GarageKit' :[7,1488],
            '3D-Mask' :[8,268],
            'Living' : [9,5000]
        }
    for i in dict :
        print(dic[i])

def printdict(dic) :
    for i in dic :
        print(i, dic[i])

def count() :
    df_all = pd.read_csv("data/train/all_image.csv")
    path_all = df_all['path_all_image']
    df_filter = pd.read_csv("data/train/filter_image.csv")
    path_filter = df_filter['path']
    score = df_filter['score']
    dic_all = {
            '2D-Display-Pad' : 0,
            '2D-Display-Phone' :0 ,
            '2D-Print-Album' :0,
            '2D-Print-Newspaper' :0,
            '2D-Print-Photo' :0,
            '2D-Print-Poster' :0,
            '3D-AdultDoll' :0,
            '3D-GarageKit' :0,
            '3D-Mask' :0, 
        }
    dic_filter = {
            '2D-Display-Pad' : 0,
            '2D-Display-Phone' :0 ,
            '2D-Print-Album' :0,
            '2D-Print-Newspaper' :0,
            '2D-Print-Photo' :0,
            '2D-Print-Poster' :0,
            '3D-AdultDoll' :0,
            '3D-GarageKit' :0,
            '3D-Mask' :0, 
        }
    spoof_all = 0
    spoof_filter = 0
    living_all = 0
    living_filter = 0
    for pt in path_all :
        if 'spoof' in pt :
           spoof_all +=1  
           for sub in dic_all :
               if sub in pt :
                   dic_all[sub] +=1
        else:
            living_all +=1
    for pt in path_filter :
        if 'spoof' in pt :
           spoof_filter +=1  
           for sub in dic_filter :
               if sub in pt :
                   dic_filter[sub] +=1
        else:
            living_filter +=1
    print(f'all image : spoof: {spoof_all}, living: {living_all}')
    printdict(dic_all)
    print(f'filter image : spoof: {spoof_filter}, living: {living_filter}')
    printdict(dic_filter)

def split_array() :
    df_filter = pd.read_csv("data/train/filter_image.csv")
    path_filter = np.array(df_filter['path'])
    spoof = []
    living = []
    for path in path_filter :
        if 'spoof' in path :
            spoof.append(path)
        else :
            living.append(path)
    
    new_spoof = np.array_split(spoof, 10)
    new_living = np.array_split(living, 10)
    for i in range(10) :
        path_save = os.path.join("data/part/", str(i)+".csv")
        new_array = np.concatenate((new_living[i], new_spoof[i]))
        np.random.shuffle(new_array)
        df = pd.DataFrame({
            'path' : new_array
        }) 
        df.to_csv(path_save, index= False)

def save_zip() :
    # for i in range(4, 10) :
    df = pd.read_csv("data/train/image_3D.csv")
    # df = pd.read_csv(os.path.join("data/part", str(i)+".csv"))
    path =df['path_image']
    zip_filename = os.path.join("/mnt/sda1/datasets/FAS-CVPR2023/train/" , "image_3D.zip")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pt in path :
            tmp = pt.split("/")
            if 'living' in pt :
                arcname = tmp[-3]+'-'+tmp[-2]+'-'+tmp[-1]
            else :
                arcname = tmp[-4] +'-'+tmp[-3]+'-'+tmp[-2]+'-'+tmp[-1]
            zipf.write(pt, arcname=arcname)
            zipf.write(pt.replace('.jpg', '.txt'), arcname=arcname.replace('.jpg', '.txt'))
    # print(f'done file {i}.zip')

def check_point() :
    (left, top), (right, bottom), dst = read_txt("data/000001.txt", rate= 1.1)
    im = cv2.imread("data/000001.jpg")
    im = cv2.rectangle(im, (left, top), (right, bottom), color= (0,0,225), thickness= 1)
    cv2.imwrite('test.jpg', im)
    print(left, top)

def parper_photo_poster():
    pathFolderSpoof = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/spoof"
    mucdich = {'Newspaper':[], 
               'Photo':[], 
               'Poster':[]
               }
    label = ['Newspaper', 'Photo', 'Poster']
    df= pd.read_csv("data/train/all_image.csv")
    path = df["path_all_image"]
    print(path)
    for dir in path :
        for i in label :
            if i in dir:
                mucdich[i].append(dir)
    newpaper = random.sample(mucdich['Newspaper'], 2000)
    photo = random.sample(mucdich['Photo'], 2000)
    poster = random.sample(mucdich['Poster'], 2000)
    zip_filename = os.path.join("/mnt/sda1/datasets/FAS-CVPR2023/train/" , "image_2D.zip")
    new_path = []
    for i in newpaper :
        new_path.append(i)
    for i in photo :
        new_path.append(i)
    for i in poster :
        new_path.append(i)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pt in new_path :
            tmp = pt.split("/")
            if 'living' in pt :
                arcname = tmp[-3]+'-'+tmp[-2]+'-'+tmp[-1]
            else :
                arcname = tmp[-4] +'-'+tmp[-3]+'-'+tmp[-2]+'-'+tmp[-1]
            zipf.write(pt, arcname=arcname)
            zipf.write(pt.replace('.jpg', '.txt'), arcname=arcname.replace('.jpg', '.txt'))

def checkdata() :
    for file in os.listdir("data/train/0") :
        if '.txt' not in file :
            path = "data/train/0/" + file
            try :
                image = cv2.imread(path)
                (left, top), (right, bottom), dst = read_txt(path.replace('.jpg', '.txt'))
                img_rate = image[top: bottom, left: right, :]
                img_rate                        =cv2.resize(img_rate, (128, 224))
            except :
                print(path)
    # path = "data/train/0/spoof-2D-Display-Phone-000572-000004.jpg"
    # (left, top), (right, bottom), dst = read_txt(path.replace('.jpg', '.txt'))
    # image = cv2.imread(path)
    # img_rate = image[abs(top): bottom, left: right, :]
    # img = cv2.rectangle(image, (left, top), (right, bottom), color= (0,0,255), thickness= 1)
    # cv2.imwrite('debug.jpg', img)

if __name__ == "__main__" :
    # x = checkReturn()
    # print(x['a'])
    # softmax()
    # save_zip()
    # count()
    # split_array()
    # save_zip()
    # check_point()
    # parper_photo_poster()
    checkdata()