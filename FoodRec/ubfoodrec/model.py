
from .userModel import UserModel
from torchvision import models
import torch
import torch.nn as nn


class UbFoodRec(nn.Module):
  # Constructor
  def __init__(self):
    super(UbFoodRec, self).__init__()
    self.resnet101 = models.resnet101(pretrained=True)
    self.resnetModel = torch.nn.Sequential(*(list(self.resnet101.children())[:-3]))
    self.AvgP = nn.AdaptiveAvgPool2d((1, 1))
    self.fcR = nn.Linear(1024, 512)
    #self.relu=nn.ReLU(inplace=True)

    self.usermodel=UserModel(input_dim = 164, output_dim = 512)

    self.fc = nn.Linear(1024, 50)

  def forward(self, x,y):
    resnet_features= self.resnetModel(x)
    resnet_features=self.AvgP(resnet_features)

    resnet_features = resnet_features.view(resnet_features.size(0), -1)
    resnet_features=self.fcR(resnet_features)
    
    user_features=self.usermodel(y)
    user_features=user_features.squeeze(1)
    
    features = torch.cat((resnet_features, user_features), dim=1)
    features= self.fc(features)

    return features
