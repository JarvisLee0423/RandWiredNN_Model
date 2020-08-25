#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       TrainerComponents.py
#   Description:        Defining the optional components for training.
#============================================================================================#

# Importing the necessary library.
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Defining the data loader.
class dataLoader():
    # Defining the method to unpickle the data file.
    @staticmethod
    def unpickle(filename):
        # Open the file.
        with open(filename, 'rb') as file:
            dict = pickle.load(file, encoding = 'bytes')
        # Return the data.
        return dict
    # Defining the method to get the ImageNet data.
    @staticmethod
    def ImageNet(batchSize):
        # Set the normalization.
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        # Get the training data.
        traingData = DataLoader(
            datasets.ImageFolder(
                root = './Datasets/ImageNet/train/',
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            ),
            batch_size = batchSize,
            num_workers = 16,
            shuffle = True,
            pin_memory = True,
            drop_last = True
        )
        # Get the validation data.
        testData = DataLoader(
            datasets.ImageFolder(
                root = './Datasets/ImageNet/val/',
                transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            ),
            batch_size = batchSize,
            num_workers = 16,
            shuffle = False, 
            pin_memory = True, 
            drop_last = False
        )
        # Return the data.
        return traingData, testData
    # Defining the method to get the MNIST data.
    @staticmethod
    def MNIST(batchSize):
        # Get the root of the dataset.
        root = './Datastes/'
        # Check whether the dataset exists or not.
        if os.path.exists(root + 'MNIST'):
            downloadStates = False
        else:
            downloadStates = True
        # Get the training data.
        traingData = DataLoader(
            dataset = datasets.MNIST(
                root = root,
                train = True,
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]),
                download = downloadStates
            ),
            batch_size = batchSize,
            shuffle = True,
            drop_last = True
        )
        # Get the test data.
        testData = DataLoader(
            dataset = datasets.MNIST(
                root = root,
                train = False,
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]),
                download = downloadStates
            ),
            batch_size = batchSize,
            shuffle = False
        )
        # Return the dataset.
        return traingData, testData
    # Defining the method to get the CIFAR10 data.
    def CIFAR10(batchSize):
        # Set the absolutely path of the raw data.
        path = "./Datasets/CIFAR10"
        # Read the data.
        data1 = dataLoader.unpickle(f'{path}/data_batch_1')
        data2 = dataLoader.unpickle(f'{path}/data_batch_2')
        data3 = dataLoader.unpickle(f'{path}/data_batch_3')
        data4 = dataLoader.unpickle(f'{path}/data_batch_4')
        data5 = dataLoader.unpickle(f'{path}/data_batch_5')
        test = dataLoader.unpickle(f'{path}/test_batch')
        # Get the data.
        ds = []
        dlabels = []
        test_ds = []
        test_dlabels = []
        for i in range(10000):
            im = np.reshape(data1[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data1[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data2[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data2[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data3[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data3[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data4[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data4[b'labels'][i])
        for i in range(10000):
            im = np.reshape(data5[b'data'][i],(3, 32, 32))
            ds.append(im)
            dlabels.append(data5[b'labels'][i])
        for i in range(10000):
            im = np.reshape(test[b'data'][i],(3, 32, 32))
            test_ds.append(im)
            test_dlabels.append(test[b'labels'][i])
        # Get the training set and test set. 
        train = torch.utils.data.TensorDataset(torch.Tensor(ds), torch.LongTensor(dlabels))
        test = torch.utils.data.TensorDataset(torch.Tensor(test_ds), torch.LongTensor(test_dlabels))
        trainingSet = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True, drop_last = True)
        testSet = torch.utils.data.DataLoader(test, batch_size = batchSize)
        # Return the data.
        return trainingSet, testSet

# Defining the Label Smoothing Regularization.
class LSR():
    # Create the function to do the LSR.
    @staticmethod
    def labelSmoothRegularization(label, numOfClasses, smoothing = 0.0):
        # Check the smoothing value.
        assert 0 <= smoothing < 1
        # Get the confidence.
        confidence = 1.0 - smoothing
        # Get the new label shape.
        labelShape = torch.Size((label.size(0), numOfClasses))
        # Without doing gradient descent.
        with torch.no_grad():
            # Get the new label.
            smoothedLabel = torch.empty(size = labelShape, device = label.device)
            smoothedLabel.fill_(smoothing / (numOfClasses - 1))
            smoothedLabel.scatter_(1, label.data.unsqueeze(1), confidence)
        # Return the smoothed label.
        return smoothedLabel

# Defining the multi-GPU training.
class MultiGPUTraining():
    # Creating the function to do the multi-GPU training.
    @staticmethod
    def DataParallelLoss(model, **kwargs):
        if 'device_ids' in kwargs.keys():
            device_ids = kwargs['device_ids']
        else:
            device_ids=None
        if 'output_device' in kwargs.keys():
            output_device = kwargs['output_device']
        else:
            output_device=None
        if 'cuda' in kwargs.keys():
            cudaID = kwargs['cuda'] 
            model = torch.nn.DataParallel(model, device_ids = device_ids, output_device = output_device).cuda(cudaID)
        else:
            model = torch.nn.DataParallel(model, device_ids = device_ids, output_device = output_device).cuda()
        return model