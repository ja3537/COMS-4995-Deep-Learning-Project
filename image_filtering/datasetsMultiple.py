import os, cv2
import numpy as np
from skimage import io, transform
import torch.nn as nn
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import random

import utils

class DatasetMultiple(Dataset):
    def __init__(self, data_root, guide1_root, guide2_root, guide3_root, is_cropped = True, is_train = True):
        self.fileID = []
        self.guide1ID = []
        self.guide2ID = []
        self.guide3ID = []
        self.data_root = data_root
        self.is_cropped = is_cropped
        self.is_train = is_train
        self.size_input = 128
        self.guide1_root = guide1_root
        self.guide2_root = guide2_root
        self.guide3_root = guide3_root
        
        for file in os.listdir(data_root):
            if file[0]!='.':
                filename = os.path.join( data_root, file )
                #                self.fileID.append(file)
                self.fileID.append(filename)

        for file in os.listdir(guide1_root):
            if file[0]!='.':
                filename = os.path.join( guide1_root, file )
                #                self.fileID.append(file)
                self.guide1ID.append(filename)
                
        for file in os.listdir(guide2_root):
            if file[0]!='.':
                filename = os.path.join( guide2_root, file )
                #                self.fileID.append(file)
                self.guide2ID.append(filename)
                
        for file in os.listdir(guide3_root):
            if file[0]!='.':
                filename = os.path.join( guide3_root, file )
                #                self.fileID.append(file)
                self.guide3ID.append(filename)

    def __getitem__(self, idx):
        filename = self.fileID[idx]
        im = io.imread(filename)
        if ( np.max(im) <= 256 ):
            im = im / 255
        else:
            print("The image format is not correct. --> %d (supposed to be in [0,255])" % (np.max(im)) )
            
        row, col, ch = im.shape
        
        if self.is_train:  
            # two kinds of noise generation model

#             scale = random.randint(5,25)
#             noise_im = np.round( (im**2.5) * 255 / scale )
#             noise_im = np.random.poisson( noise_im )
#             var = 0.001
#             gauss = utils.generateGaussNoise(noise_im, 0, var)
#             noise_im = noise_im / 255
#             noise_im = utils.validate_im( noise_im + gauss )
#             noise_im = utils.validate_im( (noise_im*scale)**0.4 )

            
            var = random.randint(100,2000)/10000
            gauss = utils.generateGaussNoise(im, 0, var)
            noise_im = utils.validate_im( im + gauss )


        guide1file = self.guide1ID[idx]
        guide1Image = io.imread(guide1file)
        guide1 = guide1Image[:,:,0]
        
        guide2file = self.guide2ID[idx]
        guide2Image = io.imread(guide2file)
        guide2 = guide2Image[:,:,0]/2 + guide2Image[:,:,1]/2
        
        guide3file = self.guide3ID[idx]
        guide3Image = io.imread(guide3file)
        guide3 = guide3Image[:,:,0]/2 + guide3Image[:,:,2]/2
        #if self.is_cropped:
            #guide = utils.modulate(guide)             # guided signal modulation
        
        inputs = np.concatenate((guide1[:,:,None], guide2[:,:,None], guide3[:,:,None]), 2)
        gt = im
        
        if self.is_cropped:
            
            h1 = random.randint(0, row - self.size_input)
            w1 = random.randint(0, col - self.size_input)
            x = inputs[ h1 : h1+self.size_input, w1 : w1+self.size_input, : ]
            y =     gt[ h1 : h1+self.size_input, w1 : w1+self.size_input, : ]
        
            rotate = random.randint(0, 3)
            if rotate != 0:
                x = np.rot90(x, rotate)
                y = np.rot90(y, rotate)
                    
            if np.random.random() >= 0.5:
                x = cv2.flip(x, flipCode=0)
                y = cv2.flip(y, flipCode=0)
        else:
            h1 = int(row/8) * 8
            w1 = int(col/8) * 8
            x = inputs[ 0:h1, 0:w1, : ]
            y =     gt[ 0:h1, 0:w1, : ]

        x = np.transpose(x,(2,0,1))
        y = np.transpose(y,(2,0,1))
        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())
        return (x, y)

    def __len__(self):
        return len(self.fileID)
