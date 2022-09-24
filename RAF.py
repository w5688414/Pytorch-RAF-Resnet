from tkinter import image_types
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnns
from torchvision import transforms
import numpy as np
import os
import pandas as pd
import torch.utils.data as data
import data_utils.raf_utils
import argparse ,random
import cv2
from GLCM.get_LBP_from_Image import LBP
from GLCM.fast_glcm import fast_glcm_entropy,fast_glcm_max,fast_glcm_ASM,fast_glcm_mean,fast_glcm_std,fast_glcm_contrast,fast_glcm_dissimilarity,fast_glcm_homogeneity
from PIL import Image

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, trans = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.trans = trans
        self.raf_path = 'datasets/raf-basic'

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [data_utils.raf_utils.flip_image,data_utils.raf_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        # print(image.shape)
        # print(image.dtype)
        #img = image.copy()
        # #get LBP
        # lbp=LBP()
        # image_array=lbp.describe(path)
        # #basic_array=lbp.lbp_basic(image_array)
        # #获取图像旋转不变LBP特征，并显示其统计直方图与特征图像
        # #revolve_array=lbp.lbp_revolve(image_array)
        # #获取图像等价模式LBP特征，并显示其统计直方图与特征图像
        # uniform_array=lbp.lbp_uniform(image_array)
        # # #获取图像旋转不变等价模式LBP特征，并显示其统计直方图与特征图像
        # #resolve_uniform_array=lbp.lbp_revolve_uniform(image_array)
        # #获取图像circular LBP特征，并显示其统计直方图与特征图像
        # #circular_array=lbp.lbp_circular(image_array)
        # imgb = uniform_array

        #get GLCM
        img=np.array(Image.open(path).convert('L'))
        mean = fast_glcm_mean(img)
        # std = fast_glcm_std(img)
        # cont = fast_glcm_contrast(img)
        # diss = fast_glcm_dissimilarity(img)
        #homo = fast_glcm_homogeneity(img)
        # asm, ene = fast_glcm_ASM(img)
        # ma = fast_glcm_max(img)
        # ent = fast_glcm_entropy(img)

        #print(image.dtype)
        #print(imgb.shape)

        # imgt = np.zeros((100,100,2))
        # imgt[:,:,0] = imgb
        # imgt[:,:,1] = mean

        #print(imgt.dtype)


        imgt = mean

        image = image[:, :,::-1]
        #imgs = imgs[:, :,::-1] # BGR to RGB
        #print(imgs[:,:,:3].shape)

        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                imgt = self.aug_func[index](imgt)
                image = self.aug_func[index](image)


        if self.transform is not None:
            # imgs = imgs.astype(np.uint8)
            image = self.transform(image)
            #print (image.dtype)
            # imgt = imgt.astype(np.uint8)
            # imgt= self.trans(imgt)
            # #print(image.shape)
            # imgs = np.zeros((4,224,224))
            # imgs[:3,:,:] = image
            # imgs[3:4,:,:] = imgt
            # imgs = imgs.astype(np.float32)

        return image, label, idx
