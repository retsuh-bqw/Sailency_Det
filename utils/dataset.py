import os
import torch
import numpy as np
import torch.nn.functional as F
from utils.preprocessing import preprocess_images, preprocess_maps



class Salicon():
    def __init__(self, path : str = './data', data:str = 'train'):
        
        imgs_train_path = os.path.join(path, 'images/train')
        maps_train_path = os.path.join(path, 'maps/train')
        imgs_val_path = os.path.join(path, 'images/val')
        maps_val_path = os.path.join(path, 'maps/val')
        if data == 'train':
            self.images = [imgs_train_path + '/' + f for f in os.listdir(imgs_train_path)]
            self.maps = [maps_train_path + '/' + f for f in os.listdir(maps_train_path) if f.endswith('.png')]
        elif data == 'val':
            self.images = [imgs_val_path + '/' + f for f in os.listdir(imgs_val_path) if f.endswith('.jpg')]
            self.maps = [maps_val_path + '/' + f for f in os.listdir(maps_val_path) if f.endswith('.png')]

        self.images.sort()
        self.maps.sort()
        

    def __getitem__(self, index):
        return self.images[index], preprocess_images(self.images[index], 480, 640), preprocess_maps(self.maps[index], 60, 80)


    def __len__(self):

        return len(self.images)