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
import argparse
from dataset.utils import read_txt, align_face
from utils.utils import str2bool
from torch.utils.data import DataLoader
from utils.utils import write_txt, save_zip
from utils import utils_config_train
from torch.utils.tensorboard import SummaryWriter

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

def ck_save_zip() :
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
    parser.add_argument('--img_input', type=str, default='img_add_img_full_aligin', choices=['img_full'\
                        ,'img_add_img_full_aligin', 'img_add_img_rate_aligin'])
    parser.add_argument('--rate', type=float, default=1.2)
    parser.add_argument('--combine', type= str, default= '016')
    
    #model
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--num_workers', default=2, type=int)
    
    
    args = parser.parse_args()
    return args

def dataset() :
    args = get_args_parser()
    from dataset.dataset_test import FasDatasetTest
    from predict import Model, pred
    from tqdm import tqdm
    import torchvision
    testDataset = FasDatasetTest(args)
    print(testDataset.__getitem__(0)['img_full_add_img_aligin'].shape)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, \
                            num_workers= args.num_workers, shuffle= False)
    model = torchvision.models.alexnet(pretrained = False)
    if args.activation == 'linear' :
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.nb_classes)
    else :
        model.classifier[-1] = nn.Sequential(
                                        nn.Linear(model.classifier[-1].in_features, 1),
                                        nn.Sigmoid()
                                        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_model = os.path.join(args.checkpoint_dir, args.name_model, args.num_train, args.num_ckpt+'.pth')
    model.load_state_dict(torch.load(checkpoint_model)['model_state_dict'])
    model.to(device)
    model.eval()
    print(model)
    # input = testDataset.__getitem__(1)['img_full_add_img_aligin'].to(device).unsqueeze(0)
    # print(testDataset.__getitem__(1)['path_image'])
    # with torch.no_grad():
    #     output = model(input)
    #     if args.activation == 'sigmoid':
    #             output = output.to('cpu').numpy()
    #     else :
    #             output = output.softmax(1).to('cpu').numpy()
    #     print(output)
    for inputs in tqdm(testLoader):
        input = inputs['img_full_add_img_aligin'].to(device)
        with torch.no_grad() : 
            output = model(input)
            if args.activation == 'sigmoid':
                output = output.to('cpu').numpy()
            else :
                output = output.softmax(1).to('cpu').numpy()
            for i in range(len(input)):
                print(output[i][-1], inputs['path_image'][i].split('/')[-1])
            break

def test_submit() :
    import shutil
    shutil.copy('results/dev/016/submit.txt', 'results')
    with open("/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/Dev.txt", 'r') as f :
        for line in f :
            write_txt('test/'+line.split()[0]+ ' ' + '0', 'results/submit.txt')

def changeThreshold(threshold, idcopy) :
    import shutil
    idpaste = len(os.listdir("results/test"))
    os.makedirs('results/test/'+str(idpaste).zfill(3))
    # shutil.copy('results/test/'+str(idcopy).zfill(3)+'/args.txt', 'results/test/'+str(idpaste).zfill(3))
    write_txt('threshold: ' + str(threshold) + '\n' + 'score idcopy: '+str(idcopy).zfill(3), 'results/test/'+str(idpaste).zfill(3)+'/args.txt')
    with open("results/test/"+str(idcopy).zfill(3)+"/submit.txt", 'r') as f :
        for line in f :
            path, score = line.split()[0], float(line.split()[1])
            if score >= threshold :
                write_txt(path+ ' ' + '1', 'results/test/'+str(idpaste).zfill(3)+'/submit.txt')
            else  :
                write_txt(path+ ' ' + '0', 'results/test/'+str(idpaste).zfill(3)+'/submit.txt')
    save_zip('results/test/'+str(idpaste).zfill(3))

def ck_data_test():
    path = []
    with open("/mnt/sda1/datasets/FAS-CVPR2023/test/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Test_V2-20230223/Test.txt", 'r') as f :
        for line in f :
            path.append(os.path.join("/mnt/sda1/datasets/FAS-CVPR2023/test/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Test_V2-20230223/data", line.split()[0]))
    # path = random.sample(path, 1000)
    zip_filename = os.path.join("data/test", 'test2000image.zip')
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i in range(1000) :
            arcname = path[i].split('/')[-1]
            zipf.write(path[i], arcname)
            zipf.write(path[i].replace('.png', '.txt'), arcname.replace('.png', '.txt'))
            
def printmodel() :
    import torch
    import torchvision.models as models

    # Tạo mô hình AlexNet
    model = models.alexnet(pretrained=True)

    # Đặt mô hình ở trạng thái đánh giá (eval)
    model.eval()

    # In ra kiến trúc của mô hình
    print(model)
    model.train()
    print(model)

def trichxuatanh():
    import torch.nn.functional as F

    # Input image tensor
    input_image = cv2.imread("data/000001.jpg")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = torch.from_numpy(input_image.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0)

    # Coordinates of eye center point
    x_eye, y_eye = 537, 646

    # Size of eye patch to extract
    w, h = 100, 100

    # Generate grid of sampling points around eye center
    grid = torch.tensor([[[x / w * 2 - 1, y / h * 2 - 1] for y in range(y_eye - h // 2, y_eye + h // 2)]
                                                        for x in range(x_eye - w // 2, x_eye + w // 2)])

    # Reshape the grid to (N, H*W, 2) where N is the batch size
    N, H, W, C = input_image.shape
    grid = grid.repeat(N, 1, 1).to(input_image.device)

    # Sample image patch around eye center using grid
    eye_patch = F.grid_sample(input_image, grid.unsqueeze(0), align_corners=True)

    print(eye_patch.squeeze_(0))
    image_np = eye_patch.squeeze_(0).cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np * 255).astype('uint8')
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('debug.jpg', image_np)

def thongke_path_image():
    path_train = []
    with open('data/txt/pathImageTrain.txt', 'r') as f:
        for line in f :
            # print(line.split()[0].split('-'))
            path_train.append(line.split()[0])
            
    return path_train

def thongke_Widt_Height() :
    # path_image = thongke_path_image()
    import pandas as pd
    sizes_face = []
    sizes_image = []
    path = []
    # Lặp qua tất cả các tệp trong thư mục
    with open('data/txt/pathImageTrain.txt', 'r') as f:
        for line in f :
            line = line.split()[0]
            image = cv2.imread(line)
            height, width, _ = image.shape 
            (left, top), (right, bottom), dst =read_txt(line.replace('.jpg', '.txt'))
            path.append(line)
            sizes_image.append((width, height))
            sizes_face.append((right- left, bottom - top))
    # Tách kích thước chiều dài và rộng thành hai danh sách riêng biệt
    df = pd.DataFrame({
        'path' : path,
        'sizes_image' : sizes_image,
        'sizes_face' : sizes_face
    })
    df.to_csv('data/csv/sizeTrain.csv', index= False)

def torchcat() :
    import torch

    # assume your Y channel has size (1, H, W)
    y_channel = torch.randn(1, 10, 10)

    # repeat y_channel along 3 channels
    x = torch.cat([y_channel, y_channel, y_channel], dim=0)
    print(x.shape)  # output: torch.Size([3, H, W])

def ketHopResults(num1, num2, rate1, rate2) :
    import shutil
    idpaste = len(os.listdir("results/test"))
    os.makedirs('results/test/'+str(idpaste).zfill(3))
    write_txt('num1: ' + str(num1) + '\n' +'num2:' + str(num2) + '\n'+ 'rate1:' + str(rate1) + '\n' + 'rate2:' + str(rate2) + '\n' , \
              'results/test/'+str(idpaste).zfill(3)+'/args.txt')
    with open("results/test/"+str(num1).zfill(3)+"/submit.txt", 'r') as f1, open("results/test/"+str(num2).zfill(3)+"/submit.txt", 'r') as f2:
        for line1, line2 in zip(f1, f2):
            path, score1, score2 = line1.split()[0], float(line1.split()[1]), float(line2.split()[1])
            score = rate1 * score1 + rate2 * score2
            write_txt(path+ ' ' + str(score), 'results/test/'+str(idpaste).zfill(3)+'/submit.txt')
    save_zip('results/test/'+str(idpaste).zfill(3))
    
def testSummaryWriter() :
    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

def config() :
    cfg = utils_config_train.config[1]
    print(cfg['LR']*0.1)
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
    # checkdata()
    # dataset()
    # test_submit()
    # changeThreshold(threshold= 0.8, idcopy= 31)
    # ck_data_test()
    # printmodel()
    # trichxuatanh()
    # thongke_path_image()
    # thongke_Widt_Height()
    # print(a)
    # torchcat()
    # ketHopResults(17, 22, 0.5, 0.5)
    # testSummaryWriter()
    config()