#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       RandWiredNN.py
#   Description:        Defining the computations of Random Wired Neural Network. 
#============================================================================================#

# Importing the necessary library.
import numpy as np
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
sys.path.append('./Random_Wired_Neural_Network_Model/')
from DAGLayer import DAGLayer
from TrainerComponents import LSR
from torch.autograd import Variable
from timeit import default_timer as timer

# Fixing the computer device and random seed.
if torch.cuda.is_available():
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Setting the computer device.
    device = 'cuda'
    # Setting the GPU listener.
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Setting the computer device.
    device = 'cpu'

# Create the class for the RWNN model.
class RandWiredNN(nn.Module):
    # Create the constructor to define all the components of the RWNN model.
    '''
        'inChannel' is the number of channels of the input data.
        'Channels' is the out channel standard when doing the training.
        'numOfClasses' is the number of classes that the image data has.
        'modelClass' is the category of the model.
            'r' for regular image model.
            's' for the small image model.
    '''
    def __init__(self, inChannel, Channels, numOfClasses, graphs, modelClass = 'r'):
        super(RandWiredNN, self).__init__()
        # Get the number of classes.
        self.numOfClasses = numOfClasses
        # Get the category of the model.
        self.modelClass = modelClass
        # Create the first convolutional layer.
        self.conv1 = nn.Conv2d(inChannel, Channels // 2, kernel_size = 3, stride = 2, padding = 1)
        # Create the first batch normalization layer.
        self.bn1 = nn.BatchNorm2d(Channels // 2)
        # Indicate whether the model class is valid.
        assert modelClass == 'r' or modelClass == 's'
        # Create the remaining convolutional layer.
        if modelClass == 'r':
            print("The model is selected as regular regmine (" + str(modelClass) + ") type.")
            self.conv2 = DAGLayer(Channels // 2, Channels, graphs[0][0], graphs[0][1])
            self.conv3 = DAGLayer(Channels, Channels * 2, graphs[1][0], graphs[1][1])
            self.conv4 = DAGLayer(Channels * 2, Channels * 4, graphs[2][0], graphs[2][1])
            self.conv5 = DAGLayer(Channels * 4, Channels * 8, graphs[3][0], graphs[3][1])
            self.convLast = nn.Conv2d(Channels * 8, 1280, kernel_size = 1)
            self.bnLast = nn.BatchNorm2d(1280)
        else:
            print("The model is selected as small regmine (" + str(modelClass) + ") type.")
            self.conv2 = nn.Conv2d(Channels // 2, Channels, kernel_size = 3, stride = 2, padding = 1)
            self.bn2 = nn.BatchNorm2d(Channels)
            self.conv3 = DAGLayer(Channels, Channels, graphs[1][0], graphs[1][1])
            self.conv4 = DAGLayer(Channels, Channels * 2, graphs[2][0], graphs[2][1])
            self.conv5 = DAGLayer(Channels * 2, Channels * 4, graphs[3][0], graphs[3][1])
            self.convLast = nn.Conv2d(Channels * 4, 1280, kernel_size = 1)
            self.bnLast = nn.BatchNorm2d(1280)
        # Create the full connection layer.
        self.fc = nn.Linear(1280, numOfClasses)
    # Create the forward propogation.
    def forward(self, x):
        # The input size should be [B, C_in, 244, 244]
        # Do the forward propogation.
        x = self.conv1(x) # [B, C_in//2, 122, 122]
        x = self.bn1(x)
        # Indicate whether the model class is valid.
        assert self.modelClass == 'r' or self.modelClass == 's'
        if self.modelClass == 's':
            x = F.relu(x)
        x = self.conv2(x) # [B, C_in, 56, 56] || [B, C_in, 56, 56]
        # Indicate whether the model class is valid.
        assert self.modelClass == 'r' or self.modelClass == 's'
        if self.modelClass == 's':
            x = self.bn2(x)
        x = self.conv3(x) # [B, C_in * 2, 28, 28] || [B, C_in, 28, 28]
        x = self.conv4(x) # [B, C_in * 4, 14, 14] || [B, C_in * 2, 14, 14]
        x = self.conv5(x) # [B, C_in * 8, 7, 7] || [B, C_in * 4, 7, 7]
        x = F.relu(x)
        x = self.convLast(x) # [B, 1280, 7, 7]
        x = self.bnLast(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # [B, 1280, 1, 1]
        # Reshape.
        x = x.view(x.size(0), -1) # [B, 1280]
        x = self.fc(x) # [B, numOfClasses]
        # Get the prediction.
        x = F.log_softmax(x, dim = -1) # [B, numOfClasses]
        # Return the output of the RWNN.
        return x
    # Defining the method for training.
    @staticmethod
    def trainer(model, optimizer, loss, scheduler, smoothing, classSize, epoches, trainSet, devSet, graphName):
        # Checking whether the model is correct or not.
        assert type(model) != type(RandWiredNN)
        # Getting the training start time.
        startTime = timer()
        # Initializing the evaluating logging list.
        evalLosses = []
        evalAccs = []
        # Initializing the training logging list.
        trainLosses = []
        trainAccs = []
        # Training the model.
        for epoch in range(epoches):
            # Getting the epoch start time.
            epochStartTime = timer()
            # Opening the file.
            logfile = open('./Training_Logging_File.txt', 'a')
            # Initializing the training loss and accuracy.
            trainLoss = []
            trainAcc = []
            # Getting the training data.
            for i, (data, label) in enumerate(trainSet):
                # Sending the data into corresponding device.
                data = Variable(data).to(device)
                label = Variable(label).to(device)
                # Computing the prediction.
                prediction = model(data)
                # Checking whether apply the label smooth regularization.
                if smoothing > 0.0:
                    # Applying the LSR.
                    smoothedLabel = LSR.labelSmoothRegularization(label, classSize, smoothing)
                else:
                    # Doing not apply the LSR.
                    smoothedLabel = label
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
                # Outputing the intermediate training information.
                if i % 100 == 0:
                    print("The iteration " + str(i) + "/" + str(len(trainSet)) + " of epoch " + str(epoch + 1) + "/" + str(epoches) + " training: Loss = " + str(cost.item()) + " || Acc = " + str(accuracy.item()) + " || GPU Memory = " + str(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024) + "/" + str(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024/ 1024))
            # Evaluating the model.
            evalLoss, evalAcc = RandWiredNN.evaluator(model.eval(), loss, smoothing, classSize, devSet)
            # Selecting the best model.
            if len(evalAccs) == 0 or evalAcc >= max(evalAccs):
                # Saving the model.
                torch.save(model.train().state_dict(), './RandWiredNN.pt')
                print("Model Saved")
                # Logging the data.
                logfile.write("Model Saved\n")
            else:
                # Applying the learning rate scheduler.
                scheduler.step()
                print("Learning Rate Decayed -> " + str(scheduler.get_last_lr()[0]))
                # Logging the data.
                logfile.write("Learning Rate Decayed -> " + str(scheduler.get_last_lr()[0]) + "\n")
            # Converting the model mode.
            model.train()
            # Getting the current time.
            currentTime = timer()
            # Storing the evaluating data.
            evalLosses.append(evalLoss)
            evalAccs.append(evalAcc)
            # Storing the training data.
            trainLosses.append(np.sum(trainLoss) / len(trainLoss))
            trainAccs.append(np.sum(trainAcc) / len(trainAcc))
            # Printing the training information.
            print("The epoch " + str(epoch + 1) + "/" + str(epoches) + " training: Loss = " + str(np.sum(trainLoss) / len(trainLoss)) + " || Acc = " + str(np.sum(trainAcc) / len(trainAcc)) + " || GPU Memory (MB) = " + str(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024) + "/" + str(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024/ 1024))
            print("The epoch " + str(epoch + 1) + "/" + str(epoches) + " evaluating: Loss = " + str(evalLoss) + " || Acc = " + str(evalAcc) + " || GPU Memory (MB) = " + str(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024) + "/" + str(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024/ 1024))
            print("The epoch " + str(epoch + 1) + "/" + str(epoches) + " consuming time (mins): " + str((currentTime - epochStartTime) / 60))
            print("The current training time (mins): " + str((currentTime - startTime) / 60))
            # Logging the data.
            logfile.write("The epoch " + str(epoch + 1) + "/" + str(epoches) + " training: Loss = " + str(np.sum(trainLoss) / len(trainLoss)) + " || Acc = " + str(np.sum(trainAcc) / len(trainAcc)) + " || GPU Memory (MB) = " + str(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024) + "/" + str(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024/ 1024) + "\n")
            logfile.write("The epoch " + str(epoch + 1) + "/" + str(epoches) + " evaluating: Loss = " + str(evalLoss) + " || Acc = " + str(evalAcc) + " || GPU Memory (MB) = " + str(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024) + "/" + str(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024/ 1024) + "\n")
            logfile.write("The epoch " + str(epoch + 1) + "/" + str(epoches) + " consuming time (mins): " + str((currentTime - epochStartTime) / 60) + "\n")
            logfile.write("The current training time (mins): " + str((currentTime - startTime) / 60) + "\n")
            # Opening the file.
            trainLossFile = open('./Training_Result_Evaluator/Training_Loss_Logging_File_' + graphName + '.txt', 'w')
            trainAccFile = open('./Training_Result_Evaluator/Training_Acc_Logging_File_' + graphName + '.txt', 'w')
            evalLossFile = open('./Training_Result_Evaluator/Evaluating_Loss_Logging_File_' + graphName + '.txt', 'w')
            evalAccFile = open('./Training_Result_Evaluator/Evaluating_Acc_Logging_File_' + graphName + '.txt', 'w')
            # Logging the training data.
            trainLossFile.write(str(trainLosses))
            trainAccFile.write(str(trainAccs))
            evalLossFile.write(str(evalLosses))
            evalAccFile.write(str(evalAccs))
            # Closing the file.
            logfile.close()
            trainLossFile.close()
            trainAccFile.close()
            evalLossFile.close()
            evalAccFile.close()
    # Defining the method for evaluating.
    @staticmethod
    def evaluator(model, loss, smoothing, classSize, devSet):
        # Initializing the evaluating loss.
        evalLoss = []
        # Initializing the evaluating accuracy.
        evalAcc = []
        # Getting the evaluating data.
        for i, (data, label) in enumerate(devSet):
            # Sending the evaluating data into corresponding device.
            data = Variable(data).to(device)
            label = Variable(label).to(device)
            # Evaluating the model.
            prediction = model(data)
            # Checking whether applying the label smooth regularization.
            if smoothing > 0.0:
                smoothedLabel = LSR.labelSmoothRegularization(label, classSize, smoothing)
            else:
                smoothedLabel = label
            # Computing the loss.
            cost = loss(prediction, smoothedLabel)
            # Storing the loss.
            evalLoss.append(cost.item())
            # Computing the accuracy.
            accuracy = (torch.argmax(prediction, 1) == label)
            accuracy = accuracy.sum().float() / len(accuracy)
            # Storing the accuracy.
            evalAcc.append(accuracy.item())
        # Returning the evaluating result.
        return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)