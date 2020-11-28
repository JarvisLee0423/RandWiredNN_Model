'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      GraphGenerator.py
    Description:    This file is used to setting the graph generator.
'''

# Importing the necessary library.
import networkx as nx
import matplotlib.pyplot as plt
from Utils.RandGraphGenerator.Generator.BAModel import BAModel
from Utils.RandGraphGenerator.Generator.ERModel import ERModel
from Utils.RandGraphGenerator.Generator.WSModel import WSModel

# The class for generating the graph data.
class Generator():
    '''
        The class for generating the graph data.
    '''
    # The function for generating the graph.
    @staticmethod
    def Generator(Cfg):
        '''
            This function is used to generating the graph.\n
            The setting of the parameters:
                'Cfg'   -The configurator of the parameters.
        '''
        # Generating the graphs.
        for i in range(4):
            # Checking whether generating the first stage.
            if i == 0:
                # Checking whether use the simple way to generating the graph.
                if Cfg.ggm == 's':
                    # Generating the BA graph.
                    if Cfg.gt == 'BA':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = BAModel.BAGeneratorSimple(Cfg.seed, Cfg.nodes // 2, Cfg.e, i)
                    # Generating the ER graph.
                    elif Cfg.gt == 'ER':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = ERModel.ERGeneratorSimple(Cfg.seed, Cfg.nodes // 2, Cfg.p, i)
                    # Generating the WS graph.
                    else:
                        # Generating the graph.
                        G, adjMatrix, edge, filename = WSModel.WSGeneratorSimple(Cfg.seed, Cfg.nodes // 2, Cfg.k, Cfg.p, i)
                else:
                    # Generating the BA graph.
                    if Cfg.gt == 'BA':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = BAModel.BAGenerator(Cfg.seed, Cfg.nodes // 2, Cfg.initialNode, Cfg.e, i)
                    # Generating the ER graph.
                    elif Cfg.gt == 'ER':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = ERModel.ERGenerator(Cfg.seed, Cfg.nodes // 2, Cfg.p, i)
                    # Generating the WS graph.
                    else:
                        # Generating the graph.
                        G, adjMatrix, edge, filename = WSModel.WSGenerator(Cfg.seed, Cfg.nodes // 2, Cfg.k, Cfg.p, i)
                # Saving the graph data.
                Generator.Save(Cfg, G, adjMatrix, edge, filename, Cfg.nodes // 2)
            else:
                # Checking whether use the simple way to generating the graph.
                if Cfg.ggm == 's':
                    # Generating the BA graph.
                    if Cfg.gt == 'BA':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = BAModel.BAGeneratorSimple(Cfg.seed, Cfg.nodes, Cfg.e, i)
                    # Generating the ER graph.
                    elif Cfg.gt == 'ER':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = ERModel.ERGeneratorSimple(Cfg.seed, Cfg.nodes, Cfg.p, i)
                    # Generating the WS graph.
                    else:
                        # Generating the graph.
                        G, adjMatrix, edge, filename = WSModel.WSGeneratorSimple(Cfg.seed, Cfg.nodes, Cfg.k, Cfg.p, i)
                else:
                    # Generating the BA graph.
                    if Cfg.gt == 'BA':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = BAModel.BAGenerator(Cfg.seed, Cfg.nodes, Cfg.initialNode, Cfg.e, i)
                    # Generating the ER graph.
                    elif Cfg.gt == 'ER':
                        # Generating the graph.
                        G, adjMatrix, edge, filename = ERModel.ERGenerator(Cfg.seed, Cfg.nodes, Cfg.p, i)
                    # Generating the WS graph.
                    else:
                        # Generating the graph.
                        G, adjMatrix, edge, filename = WSModel.WSGenerator(Cfg.seed, Cfg.nodes, Cfg.k, Cfg.p, i)
                # Saving the graph data.
                Generator.Save(Cfg, G, adjMatrix, edge, filename, Cfg.nodes)
        # Quitting the system.
        print("Generating Complete!!!")
    # The function for saving the graph data.
    @staticmethod
    def Save(Cfg, graph, adjMatrix, edge, filename, nodes):
        '''
            This function is used to saving the graph data.\n
            The setting of the parameters:
                'root'  -The root is the root for saving the graph.
                'graph' -The graph is the generated graph.
                'adjMatrix' -The adjMatrix is the graph linking data.
                'edge'      -The edge is the graph edge data.
                'filename'  -The filename of the graph data.
                'nodes'     -The number of the nodes for the graph.
        '''
        # Storing the data.
        # Reshaping the adjacent matrix into the list.
        temp = adjMatrix.reshape(nodes * nodes)
        # Opening the file.
        file = open(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/{filename}.txt', 'w')
        # Writing the data into the text file.
        count = 0
        for each in temp:
            file.write(str(each))
            file.write(' ')
            count = count + 1
            if count == nodes:
                file.write('\n')
                count = 0
        # Opening another file.
        file = open(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/{filename}_Edge.txt', 'w')
        # Storing the edge data.
        file.write(str(edge))
        # Closing the files.
        file.close()
        # Drawing the graph.
        pos = nx.shell_layout(graph)
        nx.draw(graph, pos, with_labels = True)
        plt.show()
    # The function for generating the graph data.
    @staticmethod
    def GetGraphData(root, filename):
        '''
            This function is used to getting the data of the graphs.\n
            The setting of the parameters:
                'root'      -The root of data for the graphs.
                'filename'  -The filename of data for the graphs.
        '''
        # Opening the file.
        file = open(root + f'/{filename}_Edge.txt')
        # Getting the data.
        line = file.readline()
        # Getting the data.
        edges = eval(line)
        # Opening another file.
        file = open(root + f'/{filename}.txt')
        # Getting the data.
        lines = file.readlines()
        # Getting the number of nodes.
        for line in lines:
            nodes = len(line.split(' ')) - 1
            break
        # Returning the data.
        return nodes, edges