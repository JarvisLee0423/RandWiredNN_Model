#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       DAGLayer.py
#   Description:        Defining the computations of DAG layer. 
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import sys
sys.path.append(r'./Random_Wired_Neural_Network_Model/')
from Nodes import Nodes
from collections import deque

# Fixing the random seed.
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)

# Create the class for the DAG Layer.
class DAGLayer(nn.Module):
    # Create the constructor to define all the components of the DAG layer.
    '''
        'inChannel' is the number of channels of the input data.
        'outChannel' is the number of channels of the output data.
        'numOfNode' is the number of nodes of the DAG layer.
        'edges' is the edge list of the DAG layer.
    '''
    def __init__(self, inChannel, outChannel, numOfNode, edges):
        super(DAGLayer, self).__init__()
        # Get the number of nodes for the DAG and the edge list.
        self.numOfNode = numOfNode
        self.edges = edges
        # Declare the adjacent list and the reverse adjacent list from the edges.
        # Declare the in degree list and the out degree list for each node of the DAG.
        self.adjList = []
        self.revList = []
        self.inDegree = []
        self.outDegree = []
        for _ in range(numOfNode):
            self.adjList.append([])
            self.revList.append([])
            self.inDegree.append(0)
            self.outDegree.append(0)
        # Initialize the adjacent list and the reverse adjacent list from the edges.
        # Initialize the in degree list and the out degree list for each node of the DAG.
        for i, j in edges:
            self.adjList[i].append(j)
            self.revList[j].append(i)
            self.inDegree[j] += 1
            self.outDegree[i] += 1
        # Get the input nodes and the output nodes of the DAG.
        self.inNodes = []
        self.outNodes = []
        for k in range(numOfNode):
            if self.inDegree[k] == 0:
                self.inNodes.append(k)
            if self.outDegree[k] == 0:
                self.outNodes.append(k)
        # Set the reverse adjacent list for the input nodes.
        for node in self.inNodes:
            self.revList[node].append(-1)
        # Create the model list to store the model of each node.
        self.nodes = nn.ModuleList([
            Nodes(
                inChannel = inChannel if k in self.inNodes else outChannel,
                outChannel = outChannel,
                inDegree = max(1, self.inDegree[k]),
                stride = 2 if k in self.inNodes else 1
            ) for k in range(numOfNode)
        ])
    # Create the forward propagation.
    def forward(self, x):
        # The input value of x is the size of [B, C_in, H, W].
        # Create the list to store the outputs of all the nodes.
        outputs = [None for _ in range(self.numOfNode)] + [x]
        # Create the nodes queue for the data flow.
        queue = deque(self.inNodes)
        # Get the input degree.
        inDegree = self.inDegree.copy()
        # Propogate the data inside the DAG.
        while queue:
            # Get the current node which will be used to process the input.
            currentNode = queue.popleft()
            # Get the input data list for the current node.
            inputData = [outputs[k] for k in self.revList[currentNode]]
            # Get the output data of the current node.
            # Add the inDegree info into the input data.
            outputs[currentNode] = self.nodes[currentNode](torch.stack(inputData, dim = -1))
            # When the current node's computation has been done, then decrease the inDegree of the node which has the input edge from the current node.
            for k in self.adjList[currentNode]:
                inDegree[k] -= 1
                # When the inDegree of the node k equal 0, which means that all the input data of the node k has been computed completely.
                # Therefore, add the node k into the queue.
                if inDegree[k] == 0:
                    queue.append(k)
        # Get the final output of the DAG layer.
        # The size of the output is [B, C_out, H_k, W_k], H_k and W_k is the result after doing the convolutional product of the input nodes.
        output = torch.mean(torch.stack([outputs[k] for k in self.outNodes]), dim = 0)
        # Return the output of the DAG layer.
        return output