'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      TrainerComponents.py
    Description:    This file is used to setting the trainer components.
'''

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining the trainer components.
class TrainerComponents():
    '''
        This class is used to setting the trainer components.
    '''
    # Defining the label smoothing regularization.
    @staticmethod
    def LabelSmoothRegularization(label, numOfClasses, smoothing = 0.0):
        '''
            This function is used to applying the label smooth regularization.\n
            The setting of the parameters:
                'label'         -The original label.
                'numOfClasses'  -The number of the classes.
                'smoothing'     -The value of the smoothing level.
        '''
        # Getting the confidence.
        confidence = 1.0 - smoothing
        # Getting the new label shape.
        labelShape = torch.Size((label.size(0), numOfClasses))
        # Without doing gradient descent.
        with torch.no_grad():
            # Getting the new label.
            smoothedLabel = torch.empty(size = labelShape, device = label.device)
            smoothedLabel.fill_(smoothing / (numOfClasses - 1))
            smoothedLabel.scatter_(1, label.data.unsqueeze(1), confidence)
        # Returning the smoothed label.
        return smoothedLabel
    # Defining the multi-GPU loss.
    @staticmethod
    def DataParallelLoss(model, loss, **kwargs):
        '''
            This function is used to setting the data parallel loss.\n
            The setting of the parameters:
                'model' -The model which has to be applied data parallel.
                'loss'  -The loss which has to be applied data parallel.
        '''
        # Getting the list of the GPU ID.
        if 'device_ids' in kwargs.keys():
            device_ids = kwargs['device_ids']
        else:
            device_ids = None
        # Getting the output device.
        if 'output_device' in kwargs.keys():
            output_device = kwargs['output_device']
        else:
            output_device = None
        # Checking whether the model has already been applied the cuda model.
        if 'cuda' in kwargs.keys():
            # Getting the cuda ID.
            cudaID = kwargs['cuda']
            # Setting the data parallel.
            model = nn.DataParallel(model, device_ids = device_ids, output_device = output_device).cuda(cudaID)
        else:
            model = nn.DataParallel(model, device_ids = device_ids, output_device = output_device).cuda()
        # Returning the data parallel model.
        return model