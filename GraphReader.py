'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      GraphReader.py
    Description:    This file is used to displaying the graphs.
'''

# Importing the necessary library.
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Utils.Config import Configurator

# Getting the configurator.
Cfg = Configurator.ArgParser()

# Setting the graph name.
if Cfg.gt == 'BA':
    graphName = f'{Cfg.gt}({Cfg.e})-{Cfg.seed}-'
elif Cfg.gt == 'ER':
    graphName = f'{Cfg.gt}({Cfg.p})-{Cfg.seed}-'
else:
    graphName = f'{Cfg.gt}({Cfg.k},{Cfg.p})-{Cfg.seed}-'

# Creating the graph reader.
class GraphReader():
    '''
        This class is used to displaying the graph.
    '''
    # The function is used to showing the graph.
    @staticmethod
    def DrawGraph(root, filename):
        '''
            The function is used to showing the graphs.\n
            The setting of the parameters:
                'root'      -The root of the data for graphs.
                'filename'  -The filename of the data for graphs.
        '''
        # Creating the list to contain the data.
        adjMatrix = []
        # Opening the file.
        file = open(root + f'/{filename}.txt')
        # Getting the data.
        lines = file.readlines()
        for line in lines:
            for each in line.split(' '):
                if each != '\n':
                    adjMatrix.append(eval(each))
        # Getting the number of nodes.
        for line in lines:
            nodes = len(line.split(' ')) - 1
            break
        # Getting the adjacent matrix.
        adjMatrix = np.array(adjMatrix).reshape(nodes, nodes)
        # Drawing the graph.
        G = nx.from_numpy_matrix(adjMatrix)
        # Setting the position of all the vertices.
        pos = nx.shell_layout(G)
        plt.figure(filename)
        nx.draw(G, pos, with_labels = True)
        plt.show()
        # Closing the file.
        file.close()

# Setting the main function.
if __name__ == "__main__":
    # Displaying the graphs.
    for i in range(4):
        if i == 0:
            GraphReader.DrawGraph(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{Cfg.nodes // 2}-{i}')
        else:
            GraphReader.DrawGraph(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{Cfg.nodes}-{i}')