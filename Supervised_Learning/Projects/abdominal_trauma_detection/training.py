#This is a submission to the https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data competition
#We will use AlexNet and Pytorch, in order to simultaneously learn to work with public libraries as well as doing some CV for once
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
