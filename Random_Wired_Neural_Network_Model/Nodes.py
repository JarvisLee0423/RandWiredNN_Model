#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       Nodes.py
#   Description:        Defining the computations of each node in random graph.
#   Model Description:  ->  ReLu
#                       ->  Depth-wise convolutional layer (Kernel: 3 * 3)
#                       ->  Point-wise convolutional layer (Kernel: 1 * 1)  
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixing the random seed.
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
else:
    torch.cuda.manual_seed(1)

# Create the class for the operation of each node in the DAG layer.
class Nodes(nn.Module):
    # Create the constructor to define all the components for the operation.
    '''
        'inChannel' is the number of channels of the input data.
        'outChannel' is the number of channels of the output data.
        'inDegree' is the number of input edges of the node.
        'stride' is the number of steps that the kernel will take when doing the convolutional product.
    '''
    def __init__(self, inChannel, outChannel, inDegree, stride):
        super(Nodes, self).__init__()
        # Create the weight matrix for all the input edge.
        self.weight = nn.Parameter(torch.zeros(inDegree, requires_grad = True))
        # Create the depth-wise convolutional layer.
        self.depthWise = nn.Conv2d(inChannel, inChannel, kernel_size = 3, stride = stride, padding = 1, groups = inChannel)
        # Create the point-wise convolutional layer.
        self.pointWise = nn.Conv2d(inChannel, outChannel, kernel_size = 1, stride = 1, padding = 0)
        # Create the batch normalization layer.
        self.bn = nn.BatchNorm2d(outChannel)
    # Create the forward propagation.
    def forward(self, x):
        # Multiplied the input data with the weight. [B, C_in, H, W, inDegree] -> [B, C_in, H, W]
        x = torch.matmul(x, torch.sigmoid(self.weight))
        # Do the relu.
        x = F.relu(x)
        # Do the depth wise. [B, C_in, H, W] -> [B, C_in, H_k, W_k]
        x = self.depthWise(x)
        # Do the point wise. [B, C_in, H_k, W_k] -> [B, C_out, H_k, W_k]
        x = self.pointWise(x)
        # Do the batch normalization.
        output = self.bn(x)
        # Return the output of the node.
        return output