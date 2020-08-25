#============================================================================================#
#   Copyright:          JarvisLee
#   Data:               2020/08/25
#   Project Name:       dataReader.py
#   Description:        Loading and drawing the data.
#============================================================================================#

# Importing the necessary library.
import matplotlib.pyplot as plt

# Creating the class to drawing the training result.
class dataReader():
    # Defining the method to drawing the training result.
    @staticmethod
    def drawData():
        # Getting the model name.
        file = open('./Training_Logging_File.txt', 'r')
        name = file.readline()
        name = name.split('\n')[0]
        # Open the txt file to get the data.
        file = open('./Training_Result_Evaluator/Evaluating_Acc_Logging_File_' + name + '.txt')
        evalAccs = eval(file.readline())
        file = open('./Training_Result_Evaluator/Evaluating_Loss_Logging_File_' + name + '.txt')
        evalLosses = eval(file.readline())
        file = open('./Training_Result_Evaluator/Training_Acc_Logging_File_' + name + '.txt')
        trainAccs = eval(file.readline())
        file = open('./Training_Result_Evaluator/Training_Loss_Logging_File_' + name + '.txt')
        trainLosses = eval(file.readline())
        # Close the file.
        file.close()
        # Store the corresponding data into the corresponding axis.
        x = []
        y = []
        for each in evalAccs:
            x.append(evalAccs.index(each))
            y.append(each)
        m = []
        n = []
        for each in evalLosses:
            m.append(evalLosses.index(each))
            n.append(each)
        j = []
        k = []
        for each in trainAccs:
            j.append(trainAccs.index(each))
            k.append(each)
        p = []
        q = []
        for each in trainLosses:
            p.append(trainLosses.index(each))
            q.append(each)
        # Draw the scatter plot.
        fig, _ = plt.subplots(2, 1)
        fig.tight_layout()
        plt.subplot(2, 1, 1)
        plt.title('Training Losses and Evaluation Losses')
        p1 = plt.scatter(m, n, alpha=0.6)
        p2 = plt.scatter(p, q, alpha=0.6)
        plt.legend(handles=[p1, p2], labels=['evalLoss', 'trainLoss'], loc='best')
        plt.subplot(2, 1, 2)
        plt.title('Training Accuracies and Evaluation Accuracies')
        plt.scatter(x, y, alpha=0.6)
        plt.scatter(j, k, alpha=0.6)
        plt.legend(handles=[p1, p2], labels=['evalAcc', 'trainAcc'], loc='best')
        # Annotate the max accuracy value.
        maxAccuracy = (evalAccs.index(max(evalAccs)), round(max(evalAccs), 3))
        plt.annotate('Max: ' + str(maxAccuracy), maxAccuracy)
        plt.show()

# Drawing the training result.
if __name__ == "__main__":
    # Drawing the data.
    dataReader.drawData()