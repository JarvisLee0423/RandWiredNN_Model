#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       WSModel.py
#   Description:        Generating the WS random graph.
#   Model Description:  G(k, N) where k is the number of the nearest nighbors and the N is the
#                       total number of nodes.
#============================================================================================#

# Importing the necessary library.
import networkx as nx
import numpy as np
import random

# Creating the graph generator.
class WSModel():
    # Defining the method to generating the WS graph.
    @staticmethod
    def WSGenerator(nodes = 32, k = 4, prob = 0.1, graphIndex = 0):
        '''
            'nodes' is the total number of the WS graph. (Default = 32)
            'k' is the number of the nearest neighbor. (Default = 4)
            'prob' is the rewiring probability for each edge. (Default = 0.1)
            'graphIndex' is the index of the graph. (Default = 0)
        '''
        # Getting the file name.
        filename = 'WS-Model_' + str(k) + '_' + str(prob) + '_' + str(graphIndex)
        # Getting the node list from the number of the nodes.
        nodeList = list(range(nodes))
        # Initializing the graph.
        WS = nx.Graph()
        # Initializing the adjacent matrix.
        adjMatrix = np.zeros((nodes, nodes))
        # Initializing the edge data.
        edge = []
        # Adding the node into the graph.
        WS.add_nodes_from(nodeList)
        # Initializing the WS graph.
        for i in range(nodes):
            for j in range(i + 1, k // 2 + 1 + i):
                # Avoiding out of bounds for the index.
                if j >= nodes:
                    j = j - nodes
                # Adding the edge.
                WS.add_edge(i, j)
                # Storing the edge data.
                if i < j:
                    edge.append((i, j))
                else:
                    edge.append((j, i))
                # Updating the adjacent matrix.
                adjMatrix[i][j] = 1
                adjMatrix[j][i] = 1
        # Rewiring the whole graph.
        for i in range(nodes):
            for j in range(i + 1, k // 2 + 1 + i):
                # Avoiding out if bounds for the index.
                if j >= nodes:
                    j = j - nodes
                # Getting the random number.
                r = random.random()
                # Checking whether rewiring or not.
                if r < prob:
                    # Selecting another node randomly.
                    node = np.random.randint(0, nodes)
                    # Avoiding the self loop and the overlap edge.
                    while node == i or adjMatrix[i][node] == 1:
                        # Selecting the node randomly again.
                        node = np.random.randint(0, nodes)
                    # Rewiring the graph.
                    WS.remove_edge(i, j)
                    WS.add_edge(i, node)
                    # Getting the index of the edge (i, j).
                    if i < j:
                        index = edge.index((i, j))
                    else:
                        index = edge.index((j, i))
                    # Changing the edge data.
                    if i < node:
                        edge[index] = (i, node)
                    else:
                        edge[index] = (node, i)
                    # Updating the adjacent matrix.
                    adjMatrix[i][j] = 0
                    adjMatrix[j][i] = 0
                    adjMatrix[i][node] = 1
                    adjMatrix[node][i] = 1
        # Returning the graph data.
        return WS, adjMatrix, edge, filename
    # Defining the simple method to generating the WS graph.
    @staticmethod
    def WSGeneratorSimple(nodes = 32, k = 4, prob = 0.1, graphIndex = 0):
        '''
            'nodes' is the total number of the WS graph. (Default = 32)
            'k' is the number of the nearest neighbor. (Default = 4)
            'prob' is the rewiring probability for each edge. (Default = 0.1)
            'graphIndex' is the index of the graph. (Default = 0)
        '''
        # Getting the file name.
        filename = 'WS-Model_' + str(k) + '_' + str(prob) + '_' + str(graphIndex)
        # Creating the WS graph.
        WS = nx.watts_strogatz_graph(nodes, k, prob)
        # Initializing the edge data.
        edge = []
        # Getting the adjacent matrix.
        adjMatrix = nx.to_numpy_array(WS)
        # Getting the edge data.
        for i in range(nodes):
            for j in range(nodes):
                if adjMatrix[i][j] == 1:
                    # Adding the edge.
                    if i < j:
                        # Avoiding computing the edge twice.
                        adjMatrix[j][i] = adjMatrix[i][j] = 0
                        edge.append((i, j))
                    else:
                        # Avoiding computing the edge twice.
                        adjMatrix[j][i] = adjMatrix[i][j] = 0
                        edge.append((j, i))
        # Getting the adjacent matrix again.
        adjMatrix = nx.to_numpy_array(WS)
        # Returning the graph data.
        return WS, adjMatrix, edge, filename