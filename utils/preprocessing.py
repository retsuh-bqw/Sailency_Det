import cv2
import numpy as np 
import torch
import torchvision.transforms as transforms

def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(path, shape_r, shape_c):
    ims = np.zeros((1, shape_r, shape_c, 3))

    original_image = cv2.imread(path)
    padded_image = padding(original_image, shape_r, shape_c, 3)
    ims = padded_image.astype('float')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
#     cv2 : BGR
#     PIL : RGB
    ims = ims[...,::-1]
    ims /= 255.0
    ims = np.rollaxis(ims, 2, 0)  

    ims = normalize(torch.tensor(ims.copy(), dtype=torch.float))
    return ims


def preprocess_maps(path, shape_r, shape_c):
    
    ims = np.zeros((1, 1, shape_r, shape_c))

    original_map = cv2.imread(path, 0)
    padded_map = padding(original_map, shape_r, shape_c, 1)
    ims[0, 0] = padded_map.astype(np.float32)
    ims[0, 0] /= 255.0
        

    return torch.tensor(ims.copy(), dtype=torch.float)