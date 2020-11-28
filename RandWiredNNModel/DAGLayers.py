'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      DAGLayers.py
    Description:    This file is used to setting the computations for each DAG layer.
'''

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from RandWiredNNModel.Nodes import Nodes

# Creating the class for the DAG layer.
class DAGLayers(nn.Module):
    '''
        This class is the computations for the DAG layers.\n
        Ths setting of the parameters:
            'inChannel' -The number of the channels for input data.
            'outChannel'-The number of the channels for output data.
            'numOfNodes'-The number of nodes for DAG layer.
            'edges'     -The list of the edges for DAG layer.
    '''
    # Creating the constructor.
    def __init__(self, inChannel, outChannel, numOfNodes, edges):
        # Inheritting the super class.
        super(DAGLayers, self).__init__()
        # Getting the number of nodes and the edge list for DAG layer.
        self.numOfNodes = numOfNodes
        self.edges = edges
        # Declaring the adjacent list and the reverse adjacent list from the edges.
        self.adjList = []
        self.revList = []
        # Declaring the indegree list and the outdegree list for each node of the DAG layer.
        self.inDegree = []
        self.outDegree = []
        # Initializing the above four lists.
        for _ in range(numOfNodes):
            self.adjList.append([])
            self.revList.append([])
            self.inDegree.append(0)
            self.outDegree.append(0)
        # Assigning the value for the above four lists.
        for i, j in edges:
            self.adjList[i].append(j)
            self.revList[j].append(i)
            self.inDegree[j] += 1
            self.outDegree[i] += 1
        # Getting the input and ouput nodes.
        self.inNodes = []
        self.outNodes = []
        for k in range(numOfNodes):
            if self.inDegree[k] == 0:
                self.inNodes.append(k)
            if self.outDegree[k] == 0:
                self.outNodes.append(k)
        # Resetting the reverse adjacent list for the input nodes.
        for node in self.inNodes:
            self.revList[node].append(-1)
        # Creating the model list to store the computations of each node.
        self.nodes = nn.ModuleList([
            Nodes(
                inChannel = inChannel if k in self.inNodes else outChannel,
                outChannel = outChannel,
                inDegree = max(1, self.inDegree[k]),
                stride = 2 if k in self.inNodes else 1
            ) for k in range(numOfNodes)
        ])
    # Creating the forward propagation.
    def forward(self, x):
        # The input value of x is the size of [B, C_in, H, W].
        # Creating the list to store the outputs of all the nodes and add the input data of the DAG layer at the end of the list.
        outputs = [None for _ in range(self.numOfNodes)] + [x]
        # Creating the nodes queue for the data flow.
        queue = deque(self.inNodes)
        # Getting the indegree of all the nodes in DAG layer.
        inDegree = self.inDegree.copy()
        # Propagating the data inside the DAG.
        while queue:
            # Getting the current node which will be used to process the input.
            currentNode = queue.popleft()
            # Getting all the input data of the current node.
            inputData = [outputs[k] for k in self.revList[currentNode]]
            # Getting the output data of the current node.
            outputs[currentNode] = self.nodes[currentNode](torch.stack(inputData, dim = -1))
            # When the current node's computation has been done, then decrease the inDegree of the node which has the input edge from the current node.
            for k in self.adjList[currentNode]:
                inDegree[k] -= 1
                # If the indegree of the node k becomes 0, then add the node into the queue for computation.
                if inDegree[k] == 0:
                    queue.append(k)
        # Getting the final output of the DAG layer.
        output = torch.mean(torch.stack([outputs[k] for k in self.outNodes]), dim = 0)
        # Returning the output.
        return output