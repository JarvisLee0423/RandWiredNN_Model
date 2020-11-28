'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      Trainer.py
    Description:    This file is used to training the model.
'''

# Importing the necessary library.
import os
import time
import logging
import pynvml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from visdom import Visdom
from RandWiredNNModel.RandWiredNN import RandWiredNN
from Utils.RandGraphGenerator.GraphGenerator import Generator
from Utils.Config import Configurator
from Utils.DataPreprocessor import dataLoader
from Utils.TrainerComponents import TrainerComponents

# Getting the configurator.
Cfg = Configurator.ArgParser()

# Setting the current time.
currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

# Setting the graph name.
if Cfg.gt == 'BA':
    graphName = f'{Cfg.gt}({Cfg.e})-{Cfg.seed}-'
elif Cfg.gt == 'ER':
    graphName = f'{Cfg.gt}({Cfg.p})-{Cfg.seed}-'
elif Cfg.gt == 'WS':
    graphName = f'{Cfg.gt}({Cfg.k},{Cfg.p})-{Cfg.seed}-'
else:
    graphName = f'{Cfg.gt}({Cfg.nodes})-{Cfg.seed}-'

# Creating the graph directory.
if not os.path.exists(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}'):
    os.makedirs(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}')
# Creating the model directory.
if not os.path.exists(Cfg.modelDir + f'/{Cfg.gt}/{Cfg.seed}/{currentTime}'):
    os.makedirs(Cfg.modelDir + f'/{Cfg.gt}/{Cfg.seed}/{currentTime}')
# Creating the log directory.
if not os.path.exists(Cfg.logDir + f'/{Cfg.gt}/{Cfg.seed}'):
    os.makedirs(Cfg.logDir + f'/{Cfg.gt}/{Cfg.seed}')

# Setting the device and random seed.
if torch.cuda.is_available():
    # Setting the device.
    device = 'cuda'
    # Fixing the device.
    if Cfg.GPUID > -1:
        # Fixing the device
        torch.cuda.set_device(Cfg.GPUID)
        # Getting the GPU reader.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(Cfg.GPUID)
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
else:
    # Setting the device.
    device = 'cpu'
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)

# Defining the method for evaluating.
def evaluator(model, loss, smoothing, classSize, devSet):
    '''
        This function is used to evaluating the model.\n
        The setting of the parameters:
            'model'     -The evaluating model.
            'loss'      -The evaluating loss.
            'smoothing' -The value of the smoothing.
            'classSize' -The value of the classes sizes.
            'devSet'    -The development datasets.
    '''
    # Initializing the evaluating loss.
    evalLoss = []
    # Initializing the evaluating accuracy.
    evalAcc = []
    # Getting the evaluating data.
    for i, (data, label) in enumerate(devSet):
        # Sending the evaluating data into corresponding device.
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        # Checking whether applying the label smooth regularization.
        if smoothing > 0.0:
            smoothedLabel = TrainerComponents.LabelSmoothRegularization(label, classSize, smoothing)
        else:
            smoothedLabel = label
        # Checking whether applying the data parallel.
        if Cfg.GPUID > -1:
            # Evaluating the model.
            prediction = model(data)
            # Computing the loss.
            cost = loss(prediction, smoothedLabel)
        else:
            # Computing the loss.
            cost, prediction = model(smoothedLabel, data)
        # Storing the loss.
        evalLoss.append(cost.item())
        # Computing the accuracy.
        accuracy = (torch.argmax(prediction, 1) == label)
        accuracy = accuracy.sum().float() / len(accuracy)
        # Storing the accuracy.
        evalAcc.append(accuracy.item())
    # Returning the evaluating result.
    return np.mean(evalLoss), np.mean(evalAcc)

# Defining the method for training.
def trainer(trainSet, devSet, graphs):
    '''
        This function is used to training the model.\n
        The setting of the parameters:
            'trainSet'  -The training set.
            'devSet'    -The development set.
            'graphs'    -The data of the graphs.
    '''
    # Creating the logging.
    logging.basicConfig(filename = Cfg.logDir + f'/{Cfg.gt}/{Cfg.seed}/logging-{currentTime}.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S %p')
    # Logging the training information.
    logging.info(Configurator.Displayer(Cfg))
    # Creating the visdom.
    vis = Visdom(env = f'{Cfg.gt}Model')
    # Creating the visdom.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss', 'EvaluatingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = f'Training and Evaluating Loss of {Cfg.gt} - {currentTime}'), name = 'TrainingLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'EvaluatingLoss')
    accGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingAcc', 'EvaluatingAcc'], xlabel = 'Epoches', ylabel = 'Acc', title = f'Training and Evaluating Loss {Cfg.gt} - {currentTime}'), name = 'TrainingAcc')
    vis.line(X = [0], Y = [0], win = accGraph, update = 'append', name = 'EvaluatingAcc')
    # Getting the inchannel.
    for _, (data, _) in enumerate(trainSet):
        # Creating the model.
        model = RandWiredNN(data.shape[1], Cfg.channels, Cfg.cs, Cfg.mt, graphs)
        break
    # Sending to the device.
    model.to(device)
    # Setting the optimizer.
    optimizer = optim.SGD(model.parameters(), lr = Cfg.lr, momentum = Cfg.momentum, weight_decay = Cfg.wd)
    # Setting the scheduler.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = len(trainSet), T_mult = 1, eta_min = 5e-5)
    # Setting the loss.
    if Cfg.smoothing > 0.0:
        loss = nn.KLDivLoss(reduction = "batchmean")
    else:
        loss = nn.CrossEntropyLoss()
    # Initializing the evaluating logging list.
    evalLosses = []
    evalAccs = []
    # Initializing the training logging list.
    trainLosses = []
    trainAccs = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the training loss and accuracy.
        trainLoss = []
        trainAcc = []
        # Creating the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', dynamic_ncols = True) as pbars:
            # Getting the training data.
            for i, (data, label) in enumerate(trainSet):
                # Sending the data into corresponding device.
                data = Variable(data).to(device)
                label = Variable(label).to(device)
                # Checking whether applying the label smooth regularization.
                if Cfg.smoothing > 0.0:
                    # Applying the LSR.
                    smoothedLabel = TrainerComponents.LabelSmoothRegularization(label, Cfg.cs, Cfg.smoothing)
                else:
                    # Doing not apply the LSR.
                    smoothedLabel = label
                # Checking whether applying the data paralle.
                if Cfg.GPUID > -1:
                    # Computing the prediction.
                    prediction = model(data)
                    # Computing the loss.
                    cost = loss(prediction, smoothedLabel)
                else:
                    # Applying the data parallel.
                    model = TrainerComponents.DataParallelLoss(model, loss)
                    # Computing the prediction.
                    prediction = model(data)
                    # Computing the loss.
                    cost = loss(prediction, smoothedLabel)
                # Storing the cost.
                trainLoss.append(cost.item())
                # Clearing the previous gradient.
                optimizer.zero_grad()
                # Computing the backward.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == label)
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcc.append(accuracy.item())
                # Applying the learning rate scheduler.
                scheduler.step(epoch + i / len(trainSet))
                #print(scheduler.get_last_lr()[0])

                # Updating the loading bar.
                pbars.update(1)
                # Updating the training information.
                pbars.set_postfix_str(' - Train Loss %.4f - Train Acc %.4f' % (np.mean(trainLoss), np.mean(trainAcc)))
        # Closing the loading bar.
        pbars.close()
        # Printing the hint for evaluating.
        print('Evaluating...', end = ' ')
        # Evaluating the model.
        evalLoss, evalAcc = evaluator(model.eval(), loss, Cfg.smoothing, Cfg.cs, devSet)
        if Cfg.GPUID > -1:
            # Getting the memory.
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024/ 1024
            # Printing the evaluating result.
            print(' - Eval Loss %.4f - Eval Acc %.4f - Memory %.4f' % (evalLoss, evalAcc, memory))
        else:
            # Printing the evaluating result.
            print(' - Eval Loss %.4f - Eval Acc %.4f' % (evalLoss, evalAcc))
        # Storing the training and evaluating information.
        trainLosses.append(np.mean(trainLoss))
        trainAccs.append(np.mean(trainAcc))
        evalLosses.append(evalLoss)
        evalAccs.append(evalAcc)
        if Cfg.GPUID > -1:
            # Logging the training information.
            logging.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f] || Memory: [%.4f] MB' % (epoch + 1, Cfg.epoches, np.mean(trainLoss), np.mean(trainAcc), evalLoss, evalAcc, memory))
        else:
            # Logging the training information.
            logging.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f]' % (epoch + 1, Cfg.epoches, np.mean(trainLoss), np.mean(trainAcc), evalLoss, evalAcc))
        # Drawing the graph.
        vis.line(
            X = [k for k in range(1, len(trainLosses) + 1)],
            Y = trainLosses,
            win = lossGraph,
            update = 'new',
            name = 'TrainingLoss'
        )
        vis.line(
            X = [k for k in range(1, len(evalLosses) + 1)],
            Y = evalLosses,
            win = lossGraph,
            update = 'new',
            name = 'EvaluatingLoss'
        )
        vis.line(
            X = [k for k in range(1, len(trainAccs) + 1)],
            Y = trainAccs,
            win = accGraph,
            update = 'new',
            name = 'TrainingAcc'
        )
        vis.line(
            X = [k for k in range(1, len(evalAccs) + 1)],
            Y = evalAccs,
            win = accGraph,
            update = 'new',
            name = 'EvaluatingAcc'
        )
        # Saving the model.
        torch.save(model.train().state_dict(), Cfg.modelDir + f'/{Cfg.gt}/{Cfg.seed}/{currentTime}/{graphName}{Cfg.nodes}Model.pt')
        logging.info("Model Saved")
        # Converting the model mode.
        model.train()
    # Saving the graph.
    vis.save(envs = [f'{Cfg.gt}Model'])

# Setting the main function.
if __name__ == "__main__":
    # Creating the training and development set.
    trainSet, devSet = dataLoader.CIFAR10(Cfg.dataDir, Cfg.bs)
    # Setting the graph list.
    graphs = []
    # Creating the graph.
    for i in range(4):
        # Checking whether the graph is the random graph or not.
        if Cfg.gt == 'BA' or Cfg.gt == 'WS' or Cfg.gt == 'ER':
            # Checking whether the graph has already exist or not.
            if os.path.exists(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/{graphName}{Cfg.nodes // 2}-{i}.txt') or os.path.exists(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/{graphName}{Cfg.nodes}-{i}.txt'):
                # Getting the data of the graphs.
                if i == 0:
                    graphs.append((Generator.GetGraphData(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{Cfg.nodes // 2}-{i}')))
                else:
                    graphs.append((Generator.GetGraphData(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{Cfg.nodes}-{i}')))
            else:
                # Creating the graphs.
                Generator.Generator(Cfg)
                # Clearing the graphs list.
                graphs.clear()
                # Getting the data of the graphs.
                for j in range(4):
                    # Getting the data of the graphs.
                    if j == 0:
                        graphs.append((Generator.GetGraphData(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{Cfg.nodes // 2}-{j}')))
                    else:
                        graphs.append((Generator.GetGraphData(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{Cfg.nodes}-{j}')))
                break
        else:
            # Checking whether the graph has already exist or not.
            if os.path.exists(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/{graphName}{i}.txt'):
                # Getting the data of the graphs.
                graphs.append((Generator.GetGraphData(Cfg.graphDir + f'/{Cfg.gt}/{Cfg.seed}/', f'{graphName}{i}')))
            else:
                # Giving the hint.
                print('Please checking your data of the graphs, there must be four graphs!')
                print('The name of the graph should be "GraphType(Nodes)-Seed-Index"')
                break
    # Training the model.
    assert (len(graphs) == 4), 'Training False'
    trainer(trainSet, devSet, graphs)