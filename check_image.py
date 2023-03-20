import cv2
from dataset.utils import align_face, read_txt
import numpy as np
img_full = cv2.imread('data/000001.jpg')
(left, top), (right, bottom), dst = read_txt('data/000001.txt')
img_align = align_face(img_full, dst)
img_face = img_full[top: bottom, left:right]
cv2.imwrite('test.jpg', img_face)
cv2.imwrite('testaligm.jpg', img_align)
img_full=cv2.resize(img_full, (128, 224))
img_face=cv2.resize(img_face, (128, 224))
img_align = cv2.resize(img_align, (128, 224))
img_full_add_img_align         = np.concatenate((img_full, img_align), axis= 1)
img_face_add_img_align         = np.concatenate((img_face, img_align), axis= 1)
cv2.imwrite('img_full_add_img_align.jpg', img_full_add_img_align)
cv2.imwrite('img_face_add_img_align.jpg', img_face_add_img_align)