#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       Trainer.py
#   Description:        Training the model.
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(r'./Random_Graph_Generator/Graphs/')
sys.path.append(r'./Random_Wired_Neural_Network_Model/')
from RandWiredNN import RandWiredNN
from graphReader import graphReader
from TrainerComponents import dataLoader

# Fixing the computer device and random seed.
if torch.cuda.is_available():
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Setting the device.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Setting the device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the learning rate.
learningRate = 0.1
# The value of the momentum.
momentum = 0.9
# The value of the weight decay.
weightDecay = 0.00005
# The value of the smoothing.
smoothing = 0.1
# The valur of the channels.
channels = 78
# The number of total classes.
classSize = 10
# The number of bacth sizes.
batchSize = 64
# The number of epoches.
epoches = 100
# The number of DAG layers.
dagStage = 4
# The model type.
modelType = 's'
# The random graph name.
graphName = 'WS-Model_4_0.1_'

# Training the model.
if __name__ == "__main__":
    # Opening the loging file.
    logfile = open('./Training_Logging_File.txt', 'w')
    # Getting the graph name.
    nameList = graphName.split('_')
    nameList.pop()
    name = '-'.join(nameList)
    print(name)
    # Logging the data.
    logfile.write(name + '\n')
    # Generating the graph data.
    graphs = []
    for i in range(dagStage):
        graphs.append((graphReader.generateGraphData(graphName + str(i))))
    # Generating the training data.
    trainSet, devSet = dataLoader.CIFAR10(batchSize)
    # Creating the model.
    model = RandWiredNN(3, channels, classSize, graphs, modelType)
    # Setting the optimizer.
    optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum, weight_decay = weightDecay)
    # Setting the scheduler.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = len(trainSet), T_mult = 2, eta_min = 0)
    # Setting the loss function.
    if smoothing > 0:
        # Giving the loss mode.
        print("Apply Label Smoothing")
        # Logging the data.
        logfile.write("Apply Label Smoothing\n")
        logfile.write("The training mode is " + modelType + " type\n")
        # Setting the loss with the LSR.
        loss = nn.KLDivLoss(reduction = "batchmean")
    else:
        # Giving the loss mode.
        print("No Label Smoothing")
        # Logging the data.
        logfile.write("No Label Smoothing\n")
        logfile.write("The training mode is " + modelType + " type\n")
        # Setting the loss without LSR.
        loss = nn.CrossEntropyLoss()
    # Closing the file.
    logfile.close()
    # Training the model.
    RandWiredNN.trainer(model.to(device), optimizer, loss, scheduler, smoothing, classSize, epoches, trainSet, devSet, name)