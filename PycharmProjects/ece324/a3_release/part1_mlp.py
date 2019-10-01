import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# Initialize Hyperparameters
seed = 1
learningrate = 0.05
numepoch = 100
torch.manual_seed(seed)

t_set = np.loadtxt('traindata.csv', dtype=np.single, delimiter=',')
t_label = np.loadtxt('trainlabel.csv', dtype=np.single, delimiter=',')
v_set = np.loadtxt('validdata.csv', dtype=np.single, delimiter=',')
v_label = np.loadtxt('validlabel.csv', dtype=np.single, delimiter=',')
t_set = torch.from_numpy(t_set)
t_label = torch.from_numpy(t_label)
v_set = torch.from_numpy(v_set)
v_label = torch.from_numpy(v_label)


def accuracy(predictions, label):
    total_corr = 0
    index = 0
    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1.0
        else:
            r = 0.0
        if (r == label[index].item()):
            total_corr += 1
        index += 1
    return (total_corr / len(label))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = MLP()

print("Parameter Names and Initial (random) values: ")
for name, param in net.named_parameters():
    print("name:", name, "value:", param)

predict = net(t_set)
print('accuracy:', accuracy(predict, t_label))

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learningrate)

lossRec = []
vlossRec = []
nRec = []
trainAccRec = []
validAccRec = []
for i in range(numepoch):
    optimizer.zero_grad()
    predict = net(t_set)
    loss = loss_function(input=predict.squeeze(), target=t_label.float())
    loss.backward()
    optimizer.step()
    trainAcc = accuracy(predict, t_label)
    # Computing Validation accuracy and loss
    predict = net(v_set)
    vloss = loss_function(input=predict.squeeze(), target=v_label.float())
    validAcc = accuracy(predict, v_label)

    print("loss: ", f'{loss:.4f}',"validloss",f'{vloss:.4f}', " trainAcc: ", f'{trainAcc:.4f}', " validAcc: ", f'{validAcc:.4f}')
    lossRec.append(loss)
    vlossRec.append(vloss)
    nRec.append(i)
    trainAccRec.append(trainAcc)
    validAccRec.append(validAcc)

# Plot out the loss and the accuracy, for both training and validation, vs. epoch

plt.plot(nRec, lossRec, label='Train')
plt.plot(nRec, vlossRec, label='Validation')
plt.title('Training and Validation Loss vs. epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(nRec, trainAccRec, label='Train')
plt.plot(nRec, validAccRec, label='Validation')
plt.title('Training and Validation Accuracy vs. epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


#print("Model Weights")
#for name, param in net.named_parameters():
#    print("name:", name, "value:", param)
