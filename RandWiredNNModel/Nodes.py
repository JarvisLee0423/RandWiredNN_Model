'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      Nodes.py
    Description:    This file is used to setting the computations of each node.
'''

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the class for the computations of each node in the DAG layer.
class Nodes(nn.Module):
    '''
        This class is the computations of each node in the DAG layer.\n
        The setting of the parameters:
            'inChannel' -The number of the channels for input data.
            'outChannel'-The number of the channels for output data.
            'inDegree'  -The number of the input edges for node.
            'stride'    -The number of the moving steps for convolution.
    '''
    # Creating the constructor.
    def __init__(self, inChannel, outChannel, inDegree, stride):
        # Inheritting the super class.
        super(Nodes, self).__init__()
        # Creating the weight matrix for all the input edges.
        self.aggWeight = nn.Parameter(torch.ones(inDegree, requires_grad = True))
        # Creating the depth-wise convolutional layer.
        self.depthWise = nn.Conv2d(inChannel, inChannel, kernel_size = 3, stride = stride, padding = 1, groups = inChannel)
        # Creating the point-wise convolutional layer.
        self.pointWise = nn.Conv2d(inChannel, outChannel, kernel_size = 1, stride = 1, padding = 0)
        # Creating the batch normalization layer.
        self.bn = nn.BatchNorm2d(outChannel)
    # Creating the forward propagation.
    def forward(self, x):
        # Aggregating the data from all the input edges. [B, C_in, H, W, inDegree] -> [B, C_in, H, W]
        x = torch.matmul(x, torch.tanh(self.aggWeight))
        # Applying the relu.
        x = F.relu(x)
        # Applying the depth-wise conv. [B, C_in, H, W] -> [B, C_in, H_k, W_k]
        x = self.depthWise(x)
        # Applying the point-wise conv. [B, C_in, H_k, W_k] -> [B, C_out, H_k, W_k]
        x = self.pointWise(x)
        # Applying the batch normalization.
        x = self.bn(x)
        # Returning the result.
        return x