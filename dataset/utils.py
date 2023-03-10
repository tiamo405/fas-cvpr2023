import os
import cv2
import numpy as np
import torch

from skimage import transform
dir_name = os.path.dirname(os.path.realpath(__file__))

def landmark68_to_mtcnn(landmark):
    mtcnn = np.zeros((5, 2), dtype="int")
    x,y = 0,0
    for i in range(36,42) :
        x += landmark[i][0]
        y += landmark[i][1]
    mtcnn[0] = (x//6, y//6)
    x,y = 0,0
    for i in range(42,48) :
        x += landmark[i][0]
        y += landmark[i][1]
    mtcnn[1] = (x//6, y//6)
    mtcnn[2] = landmark[30]
    mtcnn[3] = landmark[48]
    mtcnn[4] = landmark[54]
    return mtcnn

def align_face(cv_img, dst):
    """align face theo widerface
    
    Arguments:
        cv_img {arr} -- Ảnh face gốc
        dst {arr}} -- landmark 5 điểm theo mtcnn
    
    Returns:
        arr -- Ảnh face đã align
    """
    face_img = np.zeros((112, 112), dtype=np.uint8)
    # Matrix standard lanmark same wider dataset
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32)
    
    tform = transform.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img, M, (112,112), borderValue=0.0)
    return face_img

def judge_side_face(facial_landmarks):
    """Kiểm tra mặt nghiêng dựa theo landmark
    
    Arguments:
        facial_landmarks {list or arr}} -- 5 điểm landmark
    
    Returns:
        bool -- kết quả kiểm tra
             -- False : co nghieng
             -- True : khong nghieng
    """
    wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
    high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = facial_landmarks[0] - facial_landmarks[2]
    vec_B = facial_landmarks[1] - facial_landmarks[2]
    vec_C = facial_landmarks[3] - facial_landmarks[2]
    vec_D = facial_landmarks[4] - facial_landmarks[2]
    dist_A = np.linalg.norm(vec_A)
    dist_B = np.linalg.norm(vec_B)
    dist_C = np.linalg.norm(vec_C)
    dist_D = np.linalg.norm(vec_D)

    # cal rate
    # high_rate = dist_A / dist_C
    high_rate = dist_A / dist_B
    width_rate = dist_C / dist_D
    high_ratio_variance = np.fabs(high_rate - 1)  # smaller is better
    width_ratio_variance = np.fabs(width_rate - 1)

    if dist_rate < 1.3 and width_ratio_variance < 0.2 and high_ratio_variance < 0.2:
        return True, dist_rate, width_ratio_variance, high_ratio_variance
    return False, dist_rate, width_ratio_variance, high_ratio_variance

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def read_txt(path, align =False, rate = 1.2 ) :
    points = []
    with open(path, 'r') as f:
         for line in f :
              points.append((abs(int(line.strip().split()[0])), abs(int(line.strip().split()[-1]))))
    (left, top) = points[0]
    (right, bottom) = points[1]
    dst = []
    if align == False:
        for i in range(2,7) :
            dst.append(points[i])
    else :
         for i in range(2,7) :
            dst.append((points[i][0]-left, points[i][1]- top))
    # print(dst)
    # left, top, right, bottom = int(left/ rate), int( top / rate), int(right * rate), int(bottom * rate)
    return (left, top), (right, bottom), np.array(dst)

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise
if __name__ == "__main__" :
    # print("start preprocessing/utils.py")
    # path_image = "data/DOU_4453.jpg"
    # image = cv2.imread(path_image)
    # face_lmk = "checkpoints/shape_predictor_68_face_landmarks.dat"
    # predictor = dlib.shape_predictor(face_lmk)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # face_locations = face_recognition.face_locations(gray)
    # boxes = [[l, t, r, b, 1] for (t, r, b, l) in face_locations]
    # for (startX, startY, endX, endY, _) in boxes:
    #     boxes = [[l, t, r, b, 1] for (t, r, b, l) in face_locations]
    #     for (startX, startY, endX, endY, _) in boxes:
    #         rect = dlib.rectangle(startX, startY, endX, endY)
    #         shape = predictor(gray, rect)
    #         shape = shape_to_np(shape)
    #         mtcnn = landmark68_to_mtcnn(shape)
    #         # print(mtcnn)
    #         im = align_face(image, mtcnn)
    #         cv2.imwrite('test.jpg', im)
    #         print(judge_side_face(mtcnn))
    (left, top), (right, bottom), dst = read_txt("data/000001.txt")
    img = cv2.imread("data/000001.jpg")
    im = align_face(cv_img= img, dst= dst)
    cv2.imwrite('test.jpg', im)
