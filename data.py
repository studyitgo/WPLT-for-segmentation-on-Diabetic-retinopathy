# -*- coding: utf-8 -*-
"""
Created on 2022

@author: cc
"""

from torchvision.transforms import *

from torch.utils.data import Subset
import torchvision.utils as vutils
import os
import torch

import pandas as pd
from sklearn.utils import shuffle

import numpy as np

debug = False

resolution = 224
img_stats  = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

classes = ('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR')

df_train = pd.read_csv('F:\hcc\B. Disease Grading\B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv')
df_test = pd.read_csv('F:\hcc\B. Disease Grading\B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv')

x = df_train['id_code']
y = df_train['diagnosis']
z = df_train['edema']

x, y, z = shuffle(x, y, z)
# x, y, z = x.values, y.values, z.values
# _ = y.hist()

# get class stats
n_classes = int(y.max()+1)
# print(n_classes)# 5
class_weights = len(y) / df_train.groupby('diagnosis').size().values.ravel()  # we can use this to balance our loss function
class_weights *= n_classes / class_weights.sum()
print('class_weights:', class_weights.tolist())

e_classes = 3
e_class_w = len(z)/df_train.groupby('edema').size().values.ravel()

e_class_w *= e_classes/e_class_w.sum()

from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(x.values, y.values, test_size=0.20, stratify=y, random_state=42)
test_x = df_test.id_code.values

if debug:
    train_x, train_y = train_x[:128], train_y[:128]
    valid_x, valid_y = valid_x[:64], valid_y[:64]

print(train_x.shape)
print(train_y.shape)
print(valid_x.shape)
print(valid_y.shape)
print(test_x.shape)

from PIL import Image

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, path_list, targets=None, transform=None, extension='.jpg'):
        super().__init__()
        self.root = root
        self.path_list = path_list
        # self.len = len(self.path_list)
        self.targets = targets
        self.transform = transform
        self.extension = extension

        if targets is not None:
            assert len(self.path_list) == len(self.targets)
#            self.g_targets = torch.LongTensor(targets[0])
#            self.e_targets = torch.LongTensor(targets[1])
            self.targets = torch.LongTensor(targets)
            print("len(self.path_list)",len(self.path_list))
            print("targets",self.targets.shape)

    # def __len__(self):
    #     # print(len(self.path_list))# 330
    #     return self.len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.path_list[index]
        sample = Image.open(os.path.join(self.root, path+self.extension))

        if self.transform is not None:
            sample = self.transform(sample)


        #self.ones = np.zeros

        if self.targets is not None:# excutecc

            #return sample, self.g_targets[index], self.e_targets[index]


            return sample, self.targets[index]
        else:
            #return sample, torch.LongTensor([]), torch.LongTensor([])

            return sample, torch.LongTensor([])

    def __len__(self):
        # print(len(self.path_list))# 330
        return len(self.path_list)

from PIL.Image import BICUBIC

train_transform = Compose([
    Resize([resolution]*2, BICUBIC),
    ColorJitter(brightness=0.05, contrast=0.05, saturation=0.01, hue=0),
    RandomAffine(degrees=15, translate=(0.01, 0.01), scale=(1.0, 1.25), fillcolor=(0,0,0), resample=BICUBIC),
    RandomHorizontalFlip(),
#     RandomVerticalFlip(),
    ToTensor(),
    Normalize(*img_stats)
])

test_transform = Compose([
    Resize([resolution]*2, BICUBIC),
    ToTensor(),
    Normalize(*img_stats)
])

train_dataset = ImageDataset(root='F:\hcc\B. Disease Grading\B. Disease Grading/1. Original Images/a. Training Set/',
                             path_list=train_x, targets=train_y, transform=train_transform)
train_eval_dataset = ImageDataset(root='F:\hcc\B. Disease Grading\B. Disease Grading/1. Original Images/a. Training Set/',
                                  path_list=valid_x, targets=valid_y, transform=test_transform)
test_dataset = ImageDataset(root='F:\hcc\B. Disease Grading\B. Disease Grading/1. Original Images/b. Testing Set/',
                            path_list=test_x, transform=test_transform)

print(len(train_dataset), len(test_dataset), len(train_eval_dataset))

from torch.utils.data import DataLoader

train_batch_size = 32# 32
eval_batch_size = 16# 16  ## optimized for loading speed
num_workers = 0# os.cpu_count()为20
print('num_workers:', num_workers)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers, 
                          shuffle=True, drop_last=True, pin_memory=True)
# for x,y in train_loader:
#     print("x:",x.shape)
#     print('x:',y)
#     print("y",y.shape)
    #print("z",z)
# x:sample y:diagnos z:edema

test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, num_workers=num_workers, 
                         shuffle=False, drop_last=False, pin_memory=True)

eval_train_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size, num_workers=num_workers, 
                              shuffle=False, drop_last=False, pin_memory=True)
