from losses import ArcFace , Poly1CrossEntropyLoss
import torch.nn as nn
from pytorch_metric_learning import losses

def create_loss(name_loss, num_classes):
    if name_loss =='ArcFace' : return ArcFace()
    elif name_loss =='Poly1CrossEntropyLoss' : return Poly1CrossEntropyLoss(num_classes= num_classes, reduction= 'mean')
    elif name_loss == 'CrossEntropy' : return nn.CrossEntropyLoss()
    elif name_loss =='ArFace2' : return losses.ArcFaceLoss(2, embedding_size = 2, margin=28.6, scale=64)

    