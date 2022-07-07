import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import cv2
import matplotlib.pyplot as plt
import os

from iou import *


class LargeMobileNet(nn.Module):
    def __init__(self, output_channels: int = 1): #Output en noir et blanc (1 channel) pour match le mask     
        super(LargeMobileNet, self).__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier = DeepLabHead(960, output_channels)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, img_dir: str, img_name: str):
        self.eval()
        img = cv2.imread(os.path.join(img_dir, img_name), 0).reshape(1,512,512)
        input_img = torch.from_numpy(img)
        input_img= input_img.type(torch.FloatTensor)/255
        input_img = input_img.expand(3, *input_img.shape[1:])
        a = self.forward(input_img.unsqueeze(0))
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(img[0])
        plt.subplot(122)
        pred = a["out"].cpu().detach().numpy()[0][0]
        plt.imshow(pred)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.2, 0.1, f'IoU Score={iou(pred, img):.2f}', fontsize=14,
        verticalalignment='top', bbox=props)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    model = LargeMobileNet()
    model.predict("D:/Syntell/Deep Learning/Vision/inspection_via_vision/DAGM_segmentation_toy/Val/Mask/", "1_0002_label.PNG")
