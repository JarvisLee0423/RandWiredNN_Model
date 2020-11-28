'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      DataPreprocessor.py
    Description:    This file is used to setting the data preprocessor components.
'''

# Importing the necessary library.
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Defining the data loader.
class dataLoader():
    '''
        This class is used to loading the data.
    '''
    # Defining the method to unpickle the data file.
    @staticmethod
    def Unpickle(filename):
        '''
            This function is used to unpickling the data.\n
            The setting of the parameters:
                'filename'  -The root of the data which has to be unpickled.
        '''
        # Opening the file.
        with open(filename, 'rb') as file:
            dict = pickle.load(file, encoding = 'bytes')
        # Returning the data.
        return dict
    # Defining the method to get the ImageNet data.
    @staticmethod
    def ImageNet(root, batchSize):
        '''
            This function is used to loading the ImageNet datasets.\n
            The setting of the parameters:
                'root'      -The root of the data.
                'batchSize' -The size of each batch. 
        '''
        # Setting the normalization.
        normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
        # Getting the training data.
        trainData = DataLoader(
            datasets.ImageFolder(
                root = root + '/ImageNet/train/',
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            ),
            batch_size = batchSize,
            shuffle = True,
            drop_last = True
        )
        # Getting the validation data.
        valData = DataLoader(
            datasets.ImageFolder(
                root = root + '/ImageNet/val/',
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])
            ),
            batch_size = batchSize,
            shuffle = False,
            drop_last = False
        )
        # Returning the data.
        return trainData, valData
    # Defining the method to get the MNIST data.
    @staticmethod
    def MNIST(root, batchSize):
        '''
            This function is used to loading the MNIST datasets.\n
            The setting of the parameters:
                'root'      -The root of the data.
                'batchSize' -The size of each batch.
        '''
        # Checking whether the dataset exists or not.
        if os.path.exists(root + '/MNIST/'):
            download = False
        else:
            download = True
        # Getting the training data.
        trainData = DataLoader(
            datasets.MNIST(
                root = root,
                train = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1.0,))
                ]),
                download = download
            ),
            batch_size = batchSize,
            shuffle = True,
            drop_last = True
        )
        # Getting the validation data.
        valData = DataLoader(
            datasets.MNIST(
                root = root,
                train = False,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1.0,))
                ]),
                download = download
            ),
            batch_size = batchSize,
            shuffle = False
        )
        # Returning the dataset.
        return trainData, valData
    # Defining the method to get the CIFAR10 data.
    @staticmethod
    def CIFAR10(root, batchSize):
        '''
            This function is used to loading the CIFAR10 datasets.\n
            The setting of the parameters:
                'root'      -The root of the data.
                'batchSize' -The size of each batch.
        '''
        # Unpickling the data.
        data1 = dataLoader.Unpickle(f'{root}/CIFAR10/data_batch_1')
        data2 = dataLoader.Unpickle(f'{root}/CIFAR10/data_batch_2')
        data3 = dataLoader.Unpickle(f'{root}/CIFAR10/data_batch_3')
        data4 = dataLoader.Unpickle(f'{root}/CIFAR10/data_batch_4')
        data5 = dataLoader.Unpickle(f'{root}/CIFAR10/data_batch_5')
        test = dataLoader.Unpickle(f'{root}/CIFAR10/test_batch')
        # Getting the data.
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
        trainData = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True, drop_last = True)
        valData = torch.utils.data.DataLoader(test, batch_size = batchSize)
        # Return the data.
        return trainData, valData