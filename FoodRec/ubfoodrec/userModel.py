import torch
import torch.nn as nn


class UserModel(nn.Module):
  # Constructor
  def __init__(self, input_dim, output_dim):
    super(UserModel, self).__init__()
    self.fc = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
  
  #forward
  def forward(self, x):
    output = self.fc(x)
    #print(self.layer1.weight.shape)
 
    return output 
