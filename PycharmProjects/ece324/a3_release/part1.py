import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format', default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',
                    default="valid")
parser.add_argument('numtrain', help='number of training samples', type=int, default=200)
parser.add_argument('numvalid', help='number of validation samples', type=int, default=20)
parser.add_argument('-seed', help='random seed', type=int, default=1)
parser.add_argument('-learningrate', help='learning rate', type=float, default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],
                    default='linear')
parser.add_argument('-numepoch', help='number of epochs', type=int, default=50)

args = parser.parse_args()

t_set = args.trainingfile + "data.csv"
t_label = args.trainingfile + "label.csv"

print("training data file name: ", t_set)
print("training label file name: ", t_label)

v_set = args.validationfile + "data.csv"
v_label = args.validationfile + "label.csv"

print("validation data file name: ", v_set)
print("validation label file name: ", v_label)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ", args.actfunction)

t_set = np.loadtxt("traindata.csv", dtype=np.single, delimiter=',')
t_label = np.loadtxt("trainlabel.csv",dtype=np.single, delimiter=',')
v_set = np.loadtxt("validdata.csv", dtype=np.single, delimiter=',')
v_label = np.loadtxt("validlabel.csv", dtype=np.single, delimiter=',')

# Convert from numpy array to tensor
t_set = torch.from_numpy(t_set)
t_label = torch.from_numpy(t_label)
v_set = torch.from_numpy(v_set)
v_label = torch.from_numpy(v_label)

torch.manual_seed(args.seed)

def accuracy(predictions,label):
    total_corr = 0
    index = 0
    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1.0
        else:
            r = 0.0
        if (r == label[index].item()):
            total_corr += 1
        index +=1
    return (total_corr/len(label))

class SNC(nn.Module):
    def __init__(self):
        super(SNC,self).__init__()
        self.fc1 = nn.Linear(9,1)

    def forward(self,I):
        x = self.fc1(I)
        return x

smallNN = SNC()

print("Parameter Names and Initial (random) values: ")
for name, param in smallNN.named_parameters():
    print("name:",name, "value:", param)

predict = smallNN(t_set)
print('accuracy:',accuracy(predict,t_label))

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(smallNN.parameters(),lr=args.learningrate)
lossRec = []
vlossRec = []
nRec = []
trainAccRec = []
validAccRec = []
for i in range(args.numepoch):
    optimizer.zero_grad()
    predict = smallNN(t_set)
    loss = loss_function(input = predict.squeeze(),target=t_label.float())
    loss.backward()
    optimizer.step()
    trainAcc = accuracy(predict,t_label)
    # Computing Validation accuracy and loss
    predict = smallNN(v_set)
    vloss = loss_function(input=predict.squeeze(),target=v_label.float())
    validAcc = accuracy(predict, v_label)

    print("loss: ", f'{loss:.4f}', " trainAcc: ", f'{trainAcc:.4f}', " validAcc: ", f'{validAcc:.4f}')
    lossRec.append(loss)
    vlossRec.append(vloss)
    nRec.append(i)
    trainAccRec.append(trainAcc)
    validAccRec.append(validAcc)

# Plot out the loss and the accuracy, for both training and validation, vs. epoch

plt.plot(nRec,lossRec, label='Train')
plt.plot(nRec,vlossRec, label='Validation')
plt.title('Training and Validation Loss vs. epoch')
plt.xlim(0,args.numepoch)
plt.ylim(0,1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(nRec,trainAccRec, label='Train')
plt.plot(nRec,validAccRec, label='Validation')
plt.title('Training and Validation Accuracy vs. epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim(0,args.numepoch)
plt.ylim(0,1)

plt.legend()
plt.show()

print("Model Weights")
for name, param in smallNN.named_parameters():
    print("name:",name, "value:", param)
