
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import FullCNN
import time
import os
#######################################

#matlab data dir
data_dir = 'subject'

batch_size = 16
######################################################################

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
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}
    dset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=1)
                  for x in ['train']}
    dset_sizes = len(image_datasets['train'])

#    image_datasets =  datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
#    dset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,shuffle=True, num_workers=1)
#
#    dset_sizes = len(image_datasets)
    return dset_loaders, dset_sizes

use_gpu = torch.cuda.is_available()
######################################################################

def train_model(model_ft,criterion,optimizer,lr_scheduler, num_epochs= 60):

    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    # Set model to training mode
    model_ft.train(True)
    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = reloaddata()
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0
        running_corrects = 0.0
        count = 0

        # Iterate over data.
        for data in dset_loaders['train']:
            # get the inputs
            inputs, labels = data
            #labels must to be LongTensor for CrossEntropyLoss
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs),Variable(labels.cuda())

            # zero the parameter gradients
            outputs = model_ft(inputs)
            loss = criterion(outputs,labels)
            _, preds = torch.max(outputs.data, 1)
#            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print output
            count +=1
            if count%30==0 or outputs.size()[0]<batch_size:
                print('Epoch:{}: loss:{:.3f}'.format(epoch,loss.data[0]))
                train_loss.append(loss.data[0])
            # statistics

            running_loss += loss.data[0]
            #print("***")
            #print("preds", preds.shape)
            #print('labels.data', labels.data.shape)
            #print('equal', (preds == labels.data).shape)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = float(running_corrects) / float(dset_sizes)

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()
        if epoch_acc >0.99:
            break

    #save model
    model_ft.load_state_dict(best_model_wts)
    model_out_path = "./model/best.pth"
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss

######################################################################
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=20):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


######################################################################
######################################################################
# Finetuning the convnet
# ----------------------
# Load a pretrained model and reset final fully connected layer.
model_ft = FullCNN()
criterion = nn.CrossEntropyLoss()
if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()
    model_ft = torch.nn.DataParallel(model_ft,device_ids=(0, )).cuda()

optimizer = optim.SGD((model_ft.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=0.0004)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
train_loss = train_model(model_ft,criterion,optimizer, exp_lr_scheduler,num_epochs=60)
np.save('train_loss_VGG.npy',train_loss)
#######################################


