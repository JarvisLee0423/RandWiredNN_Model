# RandWiredNN_Model
 The Randomly Wired Neural Network Pytorch Implementation

 For training the model demo one:

 When training with the CIFAR10 dataset, please download the dataset from the link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

 Then uncompressed the data into the file named CIFAR10.
 
 For environments setting:

 pip install --user -r requirements.txt

 For Training:

 python Trainer.py -lr [LearningRate] -momentum [momentum] -wd [WeightDecay] -smoothing [LabelSmoothingRegularization] -channels [StandardChannels] -bs [BatchSize] -cs [ClassSize] -ep [Epoches] -seed [RandomSeed] -gpu [GPUID] -ggm [RandomGraphGeneratingMethod] -gt [RandomGraphType] -nodes [Nodes] -initialNode [BAParam] -e [BAParam] -k [WSParam] -p [WS/ERParam] -graphDir [GraphDir] -modelDir [ModelDir] -logDir [LogDir] -dataDir [DatasetsDir]