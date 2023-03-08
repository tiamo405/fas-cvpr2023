import cv2
import pandas as pd
import os
import random
def congmang(a, numa, b, numb) :
    res = []
    for i in range(numa) :
        res.append(a[i])
    for i in range(numb) :
        res.append(b[i])
    return res    
def create_csv_train(parse = 'Train'):
    path_image_spoof = []
    label_csv_2 = []
    label_csv_10 = []
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
            'living' : [9,5000]
        }
    pathFolderSpoof = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/spoof"
    pathFolderliving = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/living"
    #spoof
    print(os.listdir(pathFolderSpoof))
    for subjects in os.listdir(pathFolderSpoof) :
        lenId = min(int(dic[subjects][1]), len(os.listdir(os.path.join(pathFolderSpoof, subjects))))
        id_sub = random.sample(os.listdir(os.path.join(pathFolderSpoof, subjects)), lenId)
        for nameIdFolder in id_sub :
            for nameFile in os.listdir(os.path.join(pathFolderSpoof, subjects, nameIdFolder)) :
                if '.txt' not in nameFile :
                    path_image_spoof.append(os.path.join(pathFolderSpoof, subjects, nameIdFolder, nameFile))

    # print(path_image_spoof)

    # 2 class
    #living
    path_image_living = []
    id_sub = random.sample(os.listdir(os.path.join(pathFolderliving)), 15000) 
    for nameIdFolder in id_sub :
        for nameFile in os.listdir(os.path.join(pathFolderliving, nameIdFolder)) :
            if '.txt' not in nameFile :
                    path_image_living.append(os.path.join(pathFolderliving, nameIdFolder, nameFile))

    path_image = congmang(path_image_spoof, 1000, path_image_living, 1000)
    for i in path_image :
        if 'spoof' in i :
            label_csv_2.append(0)
        else :
            label_csv_2.append(1)  
    df = pd.DataFrame({
        'path_image' : path_image,
        'label' : label_csv_2
    })
    df.to_csv(os.path.join('data/train', 'train_2class.csv'), index= False)
    
    # 10 class
    #living
    path_image_living = []
    id_sub = random.sample(os.listdir(os.path.join(pathFolderliving)), dic['living'][1]) 
    for nameIdFolder in id_sub :
        for nameFile in os.listdir(os.path.join(pathFolderliving, nameIdFolder)) :
            if '.txt' not in nameFile :
                    path_image_living.append(os.path.join(pathFolderliving, nameIdFolder, nameFile))
    path_image = congmang(path_image_spoof, path_image_living)
    for i in range(len(path_image)) :
        for key in dic :
            if key in path_image[i] :
                # print(path_image[i], key)
                label_csv_10.append(dic[key][0])
            # else : print(path_image[i])
    df = pd.DataFrame({
        'path_image' : path_image,
        'label' : label_csv_10
    })
    df.to_csv(os.path.join('data/train', 'train_10class.csv'), index= False)
def get_all_path_image() :
    pathFolderSpoof = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/spoof"
    pathFolderliving = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/living"
    path = []
    for subjects in os.listdir(pathFolderSpoof) :
        id_sub = os.listdir(os.path.join(pathFolderSpoof, subjects))
        for nameIdFolder in id_sub :
            for nameFile in os.listdir(os.path.join(pathFolderSpoof, subjects, nameIdFolder)) :
                if '.txt' not in nameFile : 
                    path.append(os.path.join(pathFolderSpoof, subjects, nameIdFolder, nameFile))

    id_sub = os.listdir(os.path.join(pathFolderliving))       
    for nameIdFolder in id_sub :
        for nameFile in os.listdir(os.path.join(pathFolderliving, nameIdFolder)) :
            if '.txt' not in nameFile :
                    path.append(os.path.join(pathFolderliving, nameIdFolder, nameFile))
    df = pd.DataFrame({
        'path_all_image' : path
    })
    df.to_csv(os.path.join('data/train', 'all_image.csv'), index= False)
def image_3d() :
    pathFolderSpoof = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/spoof"
    path = []
    for subjects in os.listdir(pathFolderSpoof) :
        if '3D' in subjects :
            id_sub = os.listdir(os.path.join(pathFolderSpoof, subjects))
            for nameIdFolder in id_sub :
                for nameFile in os.listdir(os.path.join(pathFolderSpoof, subjects, nameIdFolder)) :
                    if '.txt' not in nameFile : 
                        path.append(os.path.join(pathFolderSpoof, subjects, nameIdFolder, nameFile))
    df = pd.DataFrame({
        'path_image' : path
    })
    df.to_csv(os.path.join('data/train', 'image_3D.csv'), index= False)
def parper_photo_poster() :
    pathFolderSpoof = "/mnt/sda1/datasets/FAS-CVPR2023/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train/spoof"
    mucdich = ['Newpaper', 'Photo', 'Poster']
    path = []
    for subjects in os.listdir(pathFolderSpoof) :
        id_sub = os.listdir(os.path.join(pathFolderSpoof, subjects))
        for nameIdFolder in id_sub :
            for nameFile in os.listdir(os.path.join(pathFolderSpoof, subjects, nameIdFolder)) :
                if '.txt' not in nameFile and nameIdFolder in mucdich : 
                    path.append(os.path.join(pathFolderSpoof, subjects, nameIdFolder, nameFile))
    print(path)
if __name__ =="__main__":
    # create_csv_train(parse= 'Train')
    # get_all_path_image()
    # image_3d()
    parper_photo_poster()