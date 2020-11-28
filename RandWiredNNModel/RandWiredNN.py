'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      RandWiredNN.py
    Description:    This file is used to setting the computations of the whole model.
'''

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from RandWiredNNModel.Nodes import Nodes
from RandWiredNNModel.DAGLayers import DAGLayers

# Creating the class for the RandWiredNN model.
class RandWiredNN(nn.Module):
    '''
        This class is used to setting the computations of the RandWiredNN model.\n
        The setting of the parameters:
            'inChannel'     -The number of the channels for input data.
            'Channels'      -The number of the default out channels for the DAG layer.
            'numOfClasses'  -The number of the classes that the datasets have.
            'modelType'     -The type of the RandWiredNN model.
                'r' for the regular regime model.
                's' for the small regime model.
            'graphs'        -The data of the graph for each DAG layer.
    '''
    # Creating the constructor.
    def __init__(self, inChannel, Channels, numOfClasses, modelType, graphs):
        # Inheritting the super class.
        super(RandWiredNN, self).__init__()
        # Getting the number of the classes.
        self.numOfClasses = numOfClasses
        # Getting the type of the model.
        self.modelType = modelType
        # Creating the first conv layer.
        self.conv1 = nn.Conv2d(inChannel, Channels // 2, kernel_size = 3, stride = 2, padding = 1)
        # Create the first batch normalization layer.
        self.bn1 = nn.BatchNorm2d(Channels // 2)
        # Creating the remaining conv layer.
        if modelType == 'r':
            self.conv2 = DAGLayers(Channels // 2, Channels, graphs[0][0], graphs[0][1])
            self.conv3 = DAGLayers(Channels, Channels * 2, graphs[1][0], graphs[1][1])
            self.conv4 = DAGLayers(Channels * 2, Channels * 4, graphs[2][0], graphs[2][1])
            self.conv5 = DAGLayers(Channels * 4, Channels * 8, graphs[3][0], graphs[3][1])
            self.convLast = nn.Conv2d(Channels * 8, 1280, kernel_size = 1)
            self.bnLast = nn.BatchNorm2d(1280)
        else:
            self.conv2 = nn.Conv2d(Channels // 2, Channels, kernel_size = 3, stride = 2, padding = 1)
            self.bn2 = nn.BatchNorm2d(Channels)
            self.conv3 = DAGLayers(Channels, Channels, graphs[1][0], graphs[1][1])
            self.conv4 = DAGLayers(Channels, Channels * 2, graphs[2][0], graphs[2][1])
            self.conv5 = DAGLayers(Channels * 2, Channels * 4, graphs[3][0], graphs[3][1])
            self.convLast = nn.Conv2d(Channels * 4, 1280, kernel_size = 1)
            self.bnLast = nn.BatchNorm2d(1280)
        # Creating the full connected layer.
        self.fc = nn.Linear(1280, numOfClasses)
    # Creating the forward propagation.
    def forward(self, x):
        # Applying the first conv layer.
        x = self.conv1(x) # [B, C_in, H, W] -> [B, C_in // 2, H // 2, W // 2]
        # Applying the batch normalization.
        x = self.bn1(x)
        # Applying the relu for the small regime model.
        if self.modelType == 's':
            x = F.relu(x)
        # Applying the second conv layer.
        x = self.conv2(x) # [B, C_in // 2, H // 2, W // 2] -> [B, C_in, H // 4, W // 4]
        # Applying the batch normalization for the small regime model.
        if self.modelType == 's':
            x = self.bn2(x)
        # Applying the third conv layer.
        x = self.conv3(x) # [B, C_in, H // 4, W // 4] -> s: [B, C_in, H // 8, W // 8] || r: [B, C_in * 2, H // 8, W // 8]
        # Applying the forth conv layer.
        x = self.conv4(x) # s: [B, C_in, H // 8, W // 8] || r: [B, C_in * 2, H // 8, W // 8] -> s: [B, C_in * 2, H // 16, W // 16] || r: [B, C_in * 4, H // 16, W // 16]
        # Applying the fifth conv layer.
        x = self.conv5(x) # s: [B, C_in * 2, H // 16, W // 16] || r: [B, C_in * 4, H // 16, W // 16] -> s: [B, C_in * 4, H // 32, W // 32] || r: [B, C_in * 8, H // 32, W // 32]
        # Applying the relu.
        x = F.relu(x)
        # Applying the last conv layer.
        x = self.convLast(x) # s: [B, C_in * 4, H // 32, W // 32] || r: [B, C_in * 8, H // 32, W // 32] -> s: [B, 1280, H // 32, W // 32] || r: [B, 1280, H // 32, W // 32]
        # Applying the batch normalization.
        x = self.bnLast(x)
        # Applying the average pooling layer.
        x = F.adaptive_avg_pool2d(x, (1, 1)) # s: [B, 1280, H // 32, W // 32] || r: [B, 1280, H // 32, W // 32] -> [B, 1280, 1, 1]
        # Reshaping the data.
        x = x.view(x.size(0), -1) # [B, 1280, 1, 1] -> [B, 1280]
        # Applying the full connected layer.
        x = self.fc(x) # [B, 1280] -> [B, numOfClasses]
        # Getting the prediction.
        x = F.log_softmax(x, dim = -1) # [B, numOfClasses] -> [B, numOfClasses]
        # Returning the result of the RandWiredNN.
        return x