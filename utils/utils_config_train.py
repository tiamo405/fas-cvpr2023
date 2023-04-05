import torch
config = {
    1: dict(
    BATCH_SIZE = 16,
    EPOCHS = 20,
    TRAIN_ON = 'ssh',
    #dir
    DATA_ROOT = 'data/train',
    CHECKPOINT_DIR = 'checkpoints',

    #model
    NAME_MODEL = 'resnet50',# resnet50, alexnet,
    NUM_CLASSES = 2,
    # ACTIVATION = 'linear',
    LR = 1e-6,
    NUM_WORKERS = 2,
    NAME_LOSS = 'Poly1CrossEntropyLoss', # 'BCEWithLogitsLoss', 'Poly1CrossEntropyLoss', 'ArcFace'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
    MOMENTUM = 0.9,
    PIN_MEMORY = True,
    #ckpt
    NUM_SAVE_CKPT = 5,
    SAVE_CKPT = True,

    #data
    RESIZE = True,
    LOAD_WIDTH = 224,
    LOAD_HEIGHT = 224,
    IMG_INPUT = 'img_face',
    #'img_face', 'img_align', 'img_full','img_full_add_img_align', 'img_face_add_img_align',\
                #  'img_face_ycbcr', 'img_align_ycbcr'

    ),
}