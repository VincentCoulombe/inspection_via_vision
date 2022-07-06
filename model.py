import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torchvision import models
import cv2
import matplotlib.pyplot as plt
import os


class LargeMobileNet(nn.Module):
    def __init__(self, output_channels: int = 1):
        super(LargeMobileNet, self).__init__()
        # model = DeepLabV3_MobileNet_V3_Large_Weights(pretrained=True, progress=True)
        model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
        model.classifier = DeepLabHead(2048, output_channels)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, img_dir: str, img_name: str):
        self.eval()
        img = cv2.imread(os.path.join(img_dir, img_name), 0).reshape(1,1,512,512)
        a = self.forward(torch.from_numpy(img).type(torch.FloatTensor)/255)
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(img[0][0])
        plt.subplot(122)
        plt.imshow(a[0].cpu().detach().numpy()[0][0])
        plt.axis('off')
        plt.show()

class CompactCNN(nn.Module):

    def __init__(self):
        super(CompactCNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 11,stride=2,padding=5) #padding = k-1/2
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32, 11,padding=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 11,padding=5)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32,64, 7,stride=2,padding=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,64, 7,padding=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64, 7,padding=3)
        self.bn6 = nn.BatchNorm2d(64)
        
        self.conv7 = nn.Conv2d(64, 128, 3,padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3,padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, 3,padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        
        self.conv10 = nn.Conv2d(128,1,1)
        self.bn10 = nn.BatchNorm2d(1)
        
        self.complayer = nn.Conv2d(128,32,1)
        self.bn11 = nn.BatchNorm2d(32)
        
        self.gmaxp = nn.MaxPool2d(128)
        self.gavgp = nn.AvgPool2d(128)
        
        self.sneuron = nn.Conv2d(66,1,1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.bn1(F.relu(self.conv1(x))) 
        x = self.bn2(F.relu(self.conv2(x))) 
        x = self.bn3(F.relu(self.conv3(x))) 

        x = self.bn4(F.relu(self.conv4(x))) 
        x = self.bn5(F.relu(self.conv5(x))) 
        x = self.bn6(F.relu(self.conv6(x))) 
        
        x = self.bn7(F.relu(self.conv7(x))) 
        x = self.bn8(F.relu(self.conv8(x))) 
        x = self.bn9(F.relu(self.conv9(x))) 
        
        segout = self.bn10(torch.tanh(self.conv10(x)))
        
        complayer = self.bn11(self.complayer(x))
        max1 = self.gmaxp(complayer)
        avg1 = self.gavgp(complayer)
        max0 = self.gmaxp(segout)
        avg0 = self.gavgp(segout)
#         print(max0.shape,avg0.shape,max1.shape,avg1.shape)
        x = torch.cat((max0,avg0,max1,avg1),dim=1)
        cscore = torch.sigmoid(self.sneuron(x))
        return segout, cscore
