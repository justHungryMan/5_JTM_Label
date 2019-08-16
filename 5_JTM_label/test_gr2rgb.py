from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from model import FullCNN
import torch.nn.functional as FUN
import time
import os
#######################################   
modelft_file = "./model/best.pth"
#matlab data dir  
data_dir = 'subject'

batch_size = 128
######################################################################

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path, ))
        return tuple_with_path

def reloaddata():
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.150, 0.150, 0.150], [0.119, 0.119, 0.119])
        ]),
        'val': transforms.Compose([
            #transforms.Scale(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.150, 0.150, 0.150], [0.119, 0.119, 0.119])
        ]),
    }
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['val']}

    dset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=False, num_workers=1)
                  for x in ['val']}    
    dset_sizes = len(image_datasets['val'])
#    image_datasets =  datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
#    dset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,shuffle=True, num_workers=1)    
#
#    dset_sizes = len(image_datasets)
    return dset_loaders, dset_sizes


for i in range(60):
    if not os.path.exists('result/action' + str(i + 1)):
        os.makedirs('result/action' + str(i + 1))

use_gpu = torch.cuda.is_available()
######################################################################
def test_model(model,criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = reloaddata()
    # Iterate over data.
    for data in dset_loaders['val']:
        # get the inputs
        
        #print(file_name)
        #filename[0:len(filename) - 4]

        inputs, labels, paths = data

        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        # forward
        outputs =  model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for data, path in zip(outputs.data, paths):
            save_path = 'result' + path[11 : len(path) - 4]
            print('save : ', save_path)
            np.save(save_path, data)


        loss = criterion(outputs, labels)   
        if cont==0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre,outputs.data.cpu()),0)
            outLabel = torch.cat((outLabel,labels.data.cpu()),0)
        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
        print('Num:',cont)
        cont +=1

    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss/dset_sizes,
                    float(running_corrects)/float(dset_sizes)))
       
    return FUN.softmax(outPre).data.numpy(), outLabel.numpy()
    
######################################################################
#torch.cuda.set_device(2)
model_ft = torch.load(modelft_file).cuda()
criterion = nn.CrossEntropyLoss().cuda()
######################################################################
outPre, outLabel = test_model(model_ft,criterion)

np.save('./model/Pre',outPre)
np.save('./model/Label',outLabel)

