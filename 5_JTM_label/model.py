
import torch.nn as nn
#import torch
#import torch.nn.functional as F
#from torch.autograd import Variable
from torchvision import models

#fine-tuning VGG19 
class FullCNN(nn.Module):
    def __init__(self):
        super(FullCNN, self).__init__()
        loadmodel = models.vgg19(pretrained=True)
        ##you can vision model
        #print(loadmodel)
        loadmodel.classifier._modules["6"] = nn.Linear(4096,60)
        self.premodel = loadmodel
        #self.relu = nn.ReLU() 
        #softmax = nn.Softmax()
        #bilinear = nn.Upsample(size=224,mode='bilinear')

    def forward(self, inputs):
        out = self.premodel(inputs)
        #out   = self.softmax(self.relu(self.premodel(self.bilinear(inputs))))
        return out

    
    
   
