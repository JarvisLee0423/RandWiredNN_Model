#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       graphDrawer.py
#   Description:        Drawing or loading the random graph.
#============================================================================================#

# Importing the necessary library.
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# Creating the graph reader.
class graphReader():
    # Defining the method to draw the graph.
    @staticmethod
    def drawGraph(filename):
        # Creating the list to contain the data.
        adjMatrix = []
        # Opening the file.
        file = open('./Random_Graph_Generator/Graphs/' + filename + '.txt')
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
        plt.figure(name)
        nx.draw(G, pos, with_labels = True)
        plt.show()
        # Closing the file.
        file.close()
    # Defining the method to generating the graph data.
    @staticmethod
    def generateGraphData(filename):
        # Opening the file.
        file = open('./Random_Graph_Generator/Graphs/' + filename + '_Edge.txt')
        # Getting the data.
        line = file.readline()
        # Getting the data.
        edges = eval(line)
        # Opening another file.
        file = open('./Random_Graph_Generator/Graphs/' + filename + '.txt')
        # Getting the data.
        lines = file.readlines()
        # Getting the number of nodes.
        for line in lines:
            nodes = len(line.split(' ')) - 1
            break
        # Returning the data.
        return nodes, edges

if __name__ == "__main__":
    # Setting the root path.
    root = './Random_Graph_Generator/Graphs/'
    # Getting the filenames.
    for name in os.listdir(root):
        # Spliting the name.
        nameList = name.split('.')
        # Poping the last expending name of the file name.
        nameList.pop()
        # Getting the preprocessed name.
        name = '.'.join(nameList)
        # Spliting the name again.
        nameList = name.split('_')
        # Skipping the non-drawing file.
        if name != 'graphDrawer' and nameList[-1] != 'Edge':
            # Getting the graph data.
            nodes, edges = graphReader.generateGraphData(name)
            # Printing the graph data.
            print("The number of nodes: " + str(nodes))
            print("The edges list: " + str(edges))
            # Drawing the graph.
            graphReader.drawGraph(name)