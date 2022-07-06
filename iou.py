import torch
import cv2
import os
import numpy as np 

def iou(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    return float(torch.sum(intersection) / torch.sum(union))


if __name__ == '__main__':
    img1 = cv2.imread(os.path.join("D:/Syntell/Deep Learning/Vision/inspection_via_vision/DAGM_segmentation_toy/Val/Mask/", "1_0002_label.PNG"), 0).reshape(1,1,512,512)
    img2 = cv2.imread(os.path.join("D:/Syntell/Deep Learning/Vision/inspection_via_vision/DAGM_segmentation_toy/Val/Mask/", "1_0002_label.PNG"), 0).reshape(1,1,512,512)
    np.set_printoptions(threshold=np.inf)
    a = iou(torch.from_numpy(img1), torch.from_numpy(img2))
    print(a)