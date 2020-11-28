'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      ERModel.py
    Description:    This file is used to setting the ER model generator.
'''

# Importing the necessary library.
import networkx as nx
import numpy as np
import random

# Creating the graph generator.
class ERModel():
    '''
        G(N, p) where N is the total number of the nodes and the p is the wiring probability.
    '''
    # Defining the method to generating the ER graph.
    @staticmethod
    def ERGenerator(seed = 0, nodes = 32, prob = 0.1, graphIndex = 0):
        '''
            This function is used to generatoring the ER graphs.\n
            The setting of the parameters:
                'seed'  -The random seed. (Default = 0)
                'nodes' -The total number of the ER Graph. (Default = 32)
                'prob'  -The wiring probability of each pair. (Default = 0.1)
                'graphIndex'    -The index of the graph. (Default = 0)
        '''
        # Getting the file name.
        filename = f'ER({prob})-{seed}-{nodes}-{graphIndex}'
        # Getting the node list from the number of the nodes.
        nodeList = list(range(nodes))
        # Initializing the graph.
        ER = nx.Graph()
        # Initializing the adjacent matrix.
        adjMatrix = np.zeros((nodes, nodes))
        # Initializing the edge data.
        edge = []
        # Creating the graph which only has the nodes.
        ER.add_nodes_from(nodeList)
        # Randomly add the edge for each pair.
        for i in range(nodes):
            for j in range(nodes):
                # Avoiding the self loop.
                if i != j:
                    # Getting the random number.
                    r = random.random()
                    # Checking whether add the edge between the pair or not.
                    if r < prob:
                        # Checking whether there is the edge between the pair.
                        if adjMatrix[i][j] == 0:
                            # Modifying the adjacent matrix.
                            adjMatrix[i][j] = 1
                            adjMatrix[j][i] = 1
                            # Adding the edge between the vertices.
                            ER.add_edge(i, j)
                            # Storing the edge data.
                            if i < j:
                                edge.append((i, j))
                            else:
                                edge.append((j, i))
        # Returning the graph data.
        return ER, adjMatrix, edge, filename
    # Defining the simple method to generating the ER graph.
    @staticmethod
    def ERGeneratorSimple(seed = 0, nodes = 32, prob = 0.1, graphIndex = 0):
        '''
            This function is used to generatoring the ER graphs.\n
            The setting of the parameters:
                'seed'  -The random seed. (Default = 0)
                'nodes' -The total number of the ER Graph. (Default = 32)
                'prob'  -The wiring probability of each pair. (Default = 0.1)
                'graphIndex'    -The index of the graph. (Default = 0)
        '''
        # Getting the file name.
        filename = f'ER({prob})-{seed}-{nodes}-{graphIndex}'
        # Creating the ER graph.
        ER = nx.erdos_renyi_graph(nodes, prob)
        # Initializing the edge data.
        edge = []
        # Getting the adjacent matrix.
        adjMatrix = nx.to_numpy_array(ER)
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
        adjMatrix = nx.to_numpy_array(ER)
        # Returning the graph data.
        return ER, adjMatrix, edge, filename