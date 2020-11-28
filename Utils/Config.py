'''
    Copyright:      JarvisLee
    Date:           2020/08/25
    File Name:      Config.py
    Description:    This file is used to setting the default value of the hyperparameters, directories and graph data.
'''

# Importing the necessary library.
import logging
import os
import argparse
from easydict import EasyDict as Config

# Creating the configurator.
Cfg = Config()

# Setting the default value of the hyperparameters.
# Setting the default value of the learning rate.
Cfg.lr = 0.1
# Setting the default value of the momentum.
Cfg.momentum = 0.9
# Setting the default value of the weight decay.
Cfg.wd = 5e-5
# Setting the default value of the smoothing.
Cfg.smoothing = 0.1
# Setting the default value of the standard channels.
Cfg.channels = 78
# Setting the default value of the batch size.
Cfg.bs = 64
# Setting the default value of the class size.
Cfg.cs = 1000
# Setting the default value of the epoches.
Cfg.epoches = 100
# Setting the default value of the random seed.
Cfg.seed = 0
# Setting the default value of the GPU ID.
Cfg.GPUID = -1
# Setting the default value of the model type.
Cfg.mt = 's' # 's' or 'r'.

# Setting the default value of the graph data.
# Setting the default value of the graph generating method.
# 's' for simple way to generating the graph.
# 'c' for complex way to generating the graph.
Cfg.ggm = 's'
# Setting the default value of the graph type.
Cfg.gt = 'BA' # 'BA' || 'ER' || 'WS' || Other graphs should be used exactly corrent name.
# Setting the default value of the number of nodes.
Cfg.nodes = 32
# Setting the default value of the number of the initial nodes.
Cfg.initialNode = -1
# Setting the default value of the number of the edge.
Cfg.e = -1
# Setting the default value of the number of the nearest neighbors.
Cfg.k = -1
# Setting the default value of the number of the wireable probability.
Cfg.p = -1

# Setting the default value of the directories.
# Setting the default value of the graph dir.
Cfg.graphDir = os.path.join('./', 'Graphs')
# Setting the default value of the model dir.
Cfg.modelDir = os.path.join('./', 'Checkpoints')
# Setting the default value of the log dir.
Cfg.logDir = os.path.join('./', 'Logs')
# Setting the default value of the dataset dir.
Cfg.dataDir = os.path.join('./', 'Datasets')

# Setting the class for argument setting.
class Configurator():
    '''
        This class is used to setting the configurator.
    '''
    # Setting the argument displayer.
    @staticmethod
    def Displayer(Cfg):
        '''
            This function is used to displaying the arguments.\n
            The setting of the parameters:
                'Cfg'   -The Cfg is the configurator.
        '''
        # Printing the items in configurator.
        info = f'''
            Learning Rate:          {Cfg.lr}
            Momentum:               {Cfg.momentum}
            Weight Decay:           {Cfg.wd}
            Smoothing:              {Cfg.smoothing}
            Standard Channels:      {Cfg.channels}
            Batch Size:             {Cfg.bs}
            Class Size:             {Cfg.cs}
            Epoches:                {Cfg.epoches}
            Random Seed:            {Cfg.seed}
            GPU ID:                 {Cfg.GPUID}
            Model Type:             {Cfg.mt}
            Graph Generate Method:  {Cfg.ggm}
            Graph Type:             {Cfg.gt}
            Graph Nodes:            {Cfg.nodes}
            Graph Initial Nodes:    {Cfg.initialNode}
            Wireabel Edge:          {Cfg.e}
            Wireable Prob:          {Cfg.p}
            Graph Nearest Neighbor: {Cfg.k}
            Graph Directory:        {Cfg.graphDir}
            Model Directory:        {Cfg.modelDir}
            Log Directory:          {Cfg.logDir}
            Data Directory:         {Cfg.dataDir}
        '''
        # Returning the information.
        return info
    # Setting the argument handler.
    @staticmethod
    def Handler(args):
        '''
            This function is used to handling the arguments.\n
            The setting of the parameters:
                'args'  -The args is the arguments dictionary.
        '''
        # Checking the number of the channels.
        assert (args['channels'] == 78 or args['channels'] == 109 or args['channels'] == 154), 'Please input the valid value of the channels, by using -h to check the arguments information.'
        # Assigning the model type.
        if args['channels'] == 78:
            args['mt'] = 's'
        else:
            args['mt'] = 'r'
        # Checking the graph type.
        assert (args['nodes'] > 3), 'Please input the valid values for the nodes, by using -h to check the arguments information.'
        # Checking the graph parameters.
        if args['gt'] == 'BA':
            assert (args['initialNode'] < args['nodes'] and args['initialNode'] > 1), 'Please input the valid value of the initial nodes, by using -h to check the arguments information.'
            assert (args['e'] > 0 and args['e'] <= args['initialNode']), 'Please input the valid value of the edge, by using -h to check the arguments information.'
        elif args['gt'] == 'ER':
            assert (args['p'] >= 0 and args['p'] <= 1), 'Please input the valid value of the prob, by using -h to check the arguments information.'
        else:
            assert (args['p'] >= 0 and args['p'] <= 1), 'Please input the valid value of the prob, by using -h to check the arguments information.'
            assert (args['k'] < args['nodes'] and args['k'] > 1), 'Please input the valid value of the k, by using -h to check the arguments information.'
        # Returning the arguments.
        return args
    # Setting the arguments' parser.
    @staticmethod
    def ArgParser():
        '''
            This function is used to parsing the arguments.
        '''
        # Getting the configurator.
        CFG = Cfg
        # Setting the arguments' parser.
        parser = argparse.ArgumentParser(description = 'Argument Parser')
        # Setting the arguments.
        parser.add_argument('-lr', '--learningRate', type = float, dest = 'lr', default = CFG.lr, help = 'The learning rate should be the float and constrain in [0, 1].')
        parser.add_argument('-momentum', '--momentum', type = float, dest = 'momentum', default = CFG.momentum, help = 'The momentum should be the float and constrain in [0, 1].')
        parser.add_argument('-wd', '--weightDecay', type = float, dest = 'wd', default = CFG.wd, help = 'The weight decay should be the float and constrain in [0, 1].')
        parser.add_argument('-smoothing', '--smoothing', type = float, dest = 'smoothing', default = CFG.smoothing, help = 'The smoothing should be the float and constrain in [0, 1]. (Other values would be seen as non-LSR)')
        parser.add_argument('-channels', '--channels', type = int, dest = 'channels', default = CFG.channels, help = 'The channels should be the integer and only access three values 78, 109 and 154.')
        parser.add_argument('-bs', '--batchSize', type = int, dest = 'bs', default = CFG.bs, help = 'The batch size should be the integer and larger than 1.')
        parser.add_argument('-cs', '--classSize', type = int, dest = 'cs', default = CFG.cs, help = 'The class size should be the integer and larger than 1.')
        parser.add_argument('-ep', '--epoches', type = int, dest = 'epoches', default = CFG.epoches, help = 'The epoches should be the integer and larger than 1.')
        parser.add_argument('-seed', '--seed', type = int, dest = 'seed', default = CFG.seed, help = 'The random seed should be the integer and larger then -1.')
        parser.add_argument('-gpu', '--GPUID', type = int, dest = 'GPUID', default = CFG.GPUID, help = 'The GPU ID should be the integer and larger than -1. (-1 means applying the Data Parallel training!)')
        parser.add_argument('-ggm', '--graphGenerateMethod', type = str, dest = 'ggm', default = CFG.ggm, help = 'The graph generate method should be the string. (s for simple way || others for complex way)')
        parser.add_argument('-gt', '--graphType', type = str, dest = 'gt', default = CFG.gt, help = 'The graph type should be the string and can only be BA, ER and WS.')
        parser.add_argument('-nodes', '--nodes', type = int, dest = 'nodes', default = CFG.nodes, help = 'The nodes should be the integer and larger than 3.')
        parser.add_argument('-initialNode', '--initialNode', type = int, dest = 'initialNode', default = CFG.initialNode, help = 'The initial nodes should be integer and smaller than nodes.')
        parser.add_argument('-e', '-edge', type = int, dest = 'e', default = CFG.e, help = 'The edge should be the integer and larger than 0.')
        parser.add_argument('-k', '--k', type = int, dest = 'k', default = CFG.k, help = 'The k should be the integer and smaller than nodes.')
        parser.add_argument('-p', '--prob', type = float, dest = 'p', default = CFG.p, help = 'The prob should be the float and constrain in [0, 1].')
        parser.add_argument('-graphDir', '--graphDir', type = str, dest = 'graphDir', default = CFG.graphDir, help = 'The graph dir should be the string.')
        parser.add_argument('-modelDir', '--modelDir', type = str, dest = 'modelDir', default = CFG.modelDir, help = 'The model dir should be the string.')
        parser.add_argument('-logDir', '--logDir', type = str, dest = 'logDir', default = CFG.logDir, help = 'The log dir should be the string.')
        parser.add_argument('-dataDir', '--dataDir', type = str, dest = 'dataDir', default = CFG.dataDir, help = 'The data dir should be the string.')
        # Parsing the argument.
        args = vars(parser.parse_args())
        # Handling the argument.
        args = Configurator.Handler(args)
        # Updating the configurator.
        CFG.update(args)
        # Returning the configurator.
        return Config(CFG)

# Testing the configurator.
if __name__ == "__main__":
    # Printing the items in configurator.
    print(Configurator.Displayer(Cfg))
    # Updating the configurator.
    Cfg = Configurator.ArgParser()
    # Printing the items in configurator.
    print(Configurator.Displayer(Cfg))