#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       graphGenerator.py
#   Description:        Generating and storing the random graph.
#============================================================================================#

# Importing the necessary library.
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append(r'./Random_Graph_Generator/Generator/')
from BAModel import BAModel
from ERModel import ERModel
from WSModel import WSModel

if __name__ == "__main__":
    # Getting the graph categories.
    Graph = input("Generating graph ('BA' || 'ER' || 'WS'): ")
    # Checking the input is valid or not.
    while Graph != 'BA' and Graph != 'ER' and Graph != 'WS':
        # Getting the graph categories.
        Graph = input("Generating graph ('BA' || 'ER' || 'WS'): ")
    # Reading the graph hyperparameters.
    file = open('./Random_Graph_Generator/graphHyperparam.txt')
    # Finding the corresponding graph hyperparameters.
    lines = file.readlines()
    for line in lines:
        # Converting the lines.
        line = eval(line)
        # Checking whether the data is correct.
        if line['Graph'] == Graph:
            print(line)
            break
    # Getting the graph index.
    graphNum = input("Inputing the generate how many graphs (Integer): ")
    # Checking whether the input is integer.
    while True:
        try:
            # Getting the graph index.
            graphNum = int(graphNum)
            # Generating the graph.
            if line['Graph'] == 'BA':
                # Getting the graph data.
                for i in range(graphNum):
                    # Generating the graph.
                    BA, adjMatrix, edge, filename = BAModel.BAGeneratorSimple(line['nodes'], line['initialNodes'], line['wireableEdge'], i)
                    # Storing the data.
                    # Reshaping the adjacent matrix into the list.
                    temp = adjMatrix.reshape(line['nodes'] * line['nodes'])
                    # Opening the file.
                    file = open('./Random_Graph_Generator/Graphs/' + filename + '.txt', 'w')
                    # Writing the data into the text file.
                    count = 0
                    for each in temp:
                        file.write(str(each))
                        file.write(' ')
                        count = count + 1
                        if count == line['nodes']:
                            file.write('\n')
                            count = 0
                    # Opening another file.
                    file = open('./Random_Graph_Generator/Graphs/' + filename + '_Edge.txt', 'w')
                    # Storing the edge data.
                    file.write(str(edge))
                    # Closing the files.
                    file.close()
                    # Drawing the graph.
                    pos = nx.shell_layout(BA)
                    nx.draw(BA, pos, with_labels = True)
                    plt.show()
            if line['Graph'] == 'ER':
                # Getting the graph data.
                for i in range(graphNum):
                    # Generating the graph.
                    ER, adjMatrix, edge, filename = ERModel.ERGeneratorSimple(line['nodes'], line['prob'], i)
                    # Storing the data.
                    # Reshaping the adjacent matrix into the list.
                    temp = adjMatrix.reshape(line['nodes'] * line['nodes'])
                    # Opening the file.
                    file = open('./Random_Graph_Generator/Graphs/' + filename + '.txt', 'w')
                    # Writing the data into the text file.
                    count = 0
                    for each in temp:
                        file.write(str(each))
                        file.write(' ')
                        count = count + 1
                        if count == line['nodes']:
                            file.write('\n')
                            count = 0
                    # Opening another file.
                    file = open('./Random_Graph_Generator/Graphs/' + filename + '_Edge.txt', 'w')
                    # Storing the edge data.
                    file.write(str(edge))
                    # Closing the files.
                    file.close()
                    # Drawing the graph.
                    pos = nx.shell_layout(ER)
                    nx.draw(ER, pos, with_labels = True)
                    plt.show()
            if line['Graph'] == 'WS':
                # Getting the graph data.
                for i in range(graphNum):
                    # Generating the graph.
                    WS, adjMatrix, edge, filename = WSModel.WSGeneratorSimple(line['nodes'], line['k'], line['prob'], i)
                    # Storing the data.
                    # Reshaping the adjacent matrix into the list.
                    temp = adjMatrix.reshape(line['nodes'] * line['nodes'])
                    # Opening the file.
                    file = open('./Random_Graph_Generator/Graphs/' + filename + '.txt', 'w')
                    # Writing the data into the text file.
                    count = 0
                    for each in temp:
                        file.write(str(each))
                        file.write(' ')
                        count = count + 1
                        if count == line['nodes']:
                            file.write('\n')
                            count = 0
                    # Opening another file.
                    file = open('./Random_Graph_Generator/Graphs/' + filename + '_Edge.txt', 'w')
                    # Storing the edge data.
                    file.write(str(edge))
                    # Closing the files.
                    file.close()
                    # Drawing the graph.
                    pos = nx.shell_layout(WS)
                    nx.draw(WS, pos, with_labels = True)
                    plt.show()
            # Quitting the system.
            print("Generating Complete!!!")
            break
        except:
            graphNum = input("Inputing the generate how many graphs (Integer): ")