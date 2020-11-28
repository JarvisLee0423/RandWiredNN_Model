'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      BAModel.py
    Description:    This file is used to setting the BA model generator.
'''

# Importing the necessary library.
import networkx as nx
import numpy as np
import random

# Creating the graph generator.
class BAModel():
    '''
        G(N, M) where N is the total number of the nodes and the M is the initial number of the nodes. (N >> M)\n
        G(E) where E is the number of the wireable nodes that the new node could connect with.
    '''
    # Defining the method to generating the BA graph.
    @staticmethod
    def BAGenerator(seed = 0, nodes = 32, initialNodes = 5, wireableEdge = 3, graphIndex = 0):
        '''
            This function is used to generating the BA graph.\n
            The setting of the parameters:
                'seed'  -The random seed. (Default = 0)
                'nodes' -The total number of the BA Graph. (Default = 32)
                'initialNodes'  -The number of initial nodes of the BA Graph. (Default = 5)
                'wireableEdge'  -The number of wireable nodes for each new node. (Default = 3)
                'graphIndex'    -The index of the graph. (Default = 0)
        '''
        # Getting the random graph file name.
        filename = f'BA({wireableEdge})-{seed}-{nodes}-{graphIndex}'
        # Initializing the probability list for each node.
        prob = []
        # Initializing the degree list for each node.
        degree = []
        # Initializing the prob and degree.
        for i in range(initialNodes):
            prob.append(0)
            degree.append(0)
        # Setting the variables to compute the total degree of the graph.
        totalDegree = 0
        # Initializing the graph.
        BA = nx.Graph()
        # Initializing the adjacent matrix.
        adjMatrix = np.zeros((nodes, nodes))
        # Initializing the edge data.
        edge = []
        # Creating the complete graph for the initial nodes.
        # Getting the node list from the number of the nodes.
        nodeList = list(range(initialNodes))
        # Creating the initial graph for BA.
        BA.add_nodes_from(nodeList)
        # Adding the edge to the initial graph.
        for i in range(initialNodes):
            # Updating the degree.
            degree[i] = initialNodes - 1
            # Updating the prob.
            prob[i] = degree[i] / ((initialNodes * (initialNodes - 1)) / 2)
            # Updating the adjacent matrix.
            for j in range(initialNodes):
                # Avoiding the self loop.
                if i != j and adjMatrix[i][j] == 0:
                    # Adding the edge between the pair.
                    BA.add_edge(i, j)
                    # Storing the edge data.
                    if i < j:
                        edge.append((i, j))
                    else:
                        edge.append((j, i))
                    # Updating the adjacent matrix.
                    adjMatrix[i][j] = 1
                    adjMatrix[j][i] = 1
                    # Updating the total degree.
                    totalDegree += 1
        # Adding the new node into the BA.
        for i in range(nodes - initialNodes):
            # Getting the number of nodes that the current graph has.
            currentNode = initialNodes + i
            # Adding the new node.
            BA.add_node(currentNode)
            # Setting the initial number of edges that the new node has.
            newEdge = 0
            # Updating the prob and degree.
            prob.append(0)
            degree.append(0)
            # Checking whether add the enough new edges for current new node.
            while newEdge < wireableEdge:
                # Adding the new edge.
                for j in range(currentNode):
                    # Getting the random number.
                    r = random.random()
                    # Checking whether add the edge between current pair.
                    if adjMatrix[currentNode][j] == 0:
                        if r < prob[j] and newEdge < wireableEdge:
                            # Adding the new edge.
                            BA.add_edge(currentNode, j)
                            # Storing the edge data.
                            if currentNode < j:
                                edge.append((currentNode, j))
                            else:
                                edge.append((j, currentNode))
                            # Updating the adjacent matrix.
                            adjMatrix[currentNode][j] = 1
                            adjMatrix[j][currentNode] = 1
                            # Updating the degree.
                            degree[currentNode] += 1
                            degree[j] += 1
                            # Updating the total degree.
                            totalDegree += 1
                            # Updating the prob.
                            prob = list(np.array(degree) / totalDegree)
                            # Updating the number of new edges.
                            newEdge += 1
        # Returning the graph data.
        return BA, adjMatrix, edge, filename
    # Defining the simple method to generating the BA graph.
    @staticmethod
    def BAGeneratorSimple(seed = 0, nodes = 32, wireableEdge = 3, graphIndex = 0):
        '''
            This function is used to generating the BA graph.\n
            The setting of the parameters:
                'seed'  -The random seed. (Default = 0)
                'nodes' -The total number of the BA Graph. (Default = 32)
                'initialNodes'  -The number of initial nodes of the BA Graph. (Default = 5)
                'wireableEdge'  -The number of wireable nodes for each new node. (Default = 3)
                'graphIndex'    -The index of the graph. (Default = 0)
        '''
        # Getting the file name.
        filename = f'BA({wireableEdge})-{seed}-{nodes}-{graphIndex}'
        # Creating the BA graph.
        BA = nx.barabasi_albert_graph(nodes, wireableEdge)
        # Initializing the edge data.
        edge = []
        # Getting the adjacent matrix.
        adjMatrix = nx.to_numpy_array(BA)
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
        adjMatrix = nx.to_numpy_array(BA)
        # Returning the graph data.
        return BA, adjMatrix, edge, filename