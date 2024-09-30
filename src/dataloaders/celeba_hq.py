
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import os



def get_celeba(args, get_train_sampler = False, transform_train = True, crop_size = 224, drop_last=False):

    transforms_augs = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])

    transforms_no_augs = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if transform_train:
        train_transforms = transforms_augs
    else:
        train_transforms = transforms_no_augs
    
    test_transforms = transforms_no_augs

    data_dir = f'{args.root}/CelebA_HQ_facial_identity_dataset'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)

    dataset_labels = train_dataset.classes
    num_classes = len(dataset_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=  transform_train,
                                                   num_workers=args.workers, pin_memory=True, sampler=None,
                                                   persistent_workers=args.workers > 0,drop_last=drop_last)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True, sampler=None,
                                                  persistent_workers=args.workers > 0,drop_last=drop_last)
    
    if(get_train_sampler):
        return train_dataloader, None, test_dataloader, None, num_classes, dataset_labels
    
    return train_dataloader, None, test_dataloader, num_classes, dataset_labels




if __name__ == '__main__':

    pass