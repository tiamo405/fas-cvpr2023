import torch
config = {
    1: dict(
    SAVE_TXT = True,
    PHASE = 'test',
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    #dir
    PATH_DATA = '/mnt/sda1/datasets/FAS-CVPR2023/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/data',
    PATH_SAVE = 'results',
    PATH_TXT = 'data/dev/dev.txt',
    CHECKPOINT_DIR = 'checkpoints',

    #model
    NAME_MODEL = 'resnet50',# resnet50, alexnet,
    NUM_CLASSES = 2,
    THRESHOLD = 1,

    #ckpt
    NUM_TRAIN = '0009',
    NUM_CKPT = '4',

    #data
    RESIZE = True,
    LOAD_WIDTH = 224,
    LOAD_HEIGHT = 224,
    IMG_INPUT = 'img_face',
    #'img_face', 'img_align', 'img_full','img_full_add_img_align', 'img_face_add_img_align',\
                #  'img_face_ycbcr', 'img_align_ycbcr'
    BATCH_SIZE = 16,
    NUM_WORKERS = 2,
    COMBINE = '016',
    SHUFFLE = False,
    ),
}