import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
import warnings
warnings.filterwarnings("ignore")


class Mymodel(nn.Module):
    def __init__(self,
                 hidden1=70, hidden2=155, hidden3=85, hidden4=175, hidden5=20, hidden6=95,
                 RC1=0, RC2=0, RC3=1, RC4=1, RC5=0,
                 dropout1=0.05, dropout2=0.25, dropout3=0.0
                 ):
        super(Mymodel, self).__init__()
        self.fc1_1 = nn.Linear(13, hidden1)
        self.fc1_2 = nn.Linear(hidden1, hidden2)
        self.fc1_3 = nn.Linear(hidden2, hidden2)
        self.fc1_4 = nn.Linear(hidden2, hidden3)
        self.fc1_5 = nn.Linear(hidden3, hidden3)
        self.fc1_6 = nn.Linear(hidden3, 1)
        # Remove f2 and try to add the second part of the residual connection to see if it has an effect
        self.fc2_1 = nn.Linear(13, hidden4)
        self.fc2_2 = nn.Linear(hidden4, hidden5)
        self.fc2_3 = nn.Linear(hidden5, hidden5)
        self.fc2_4 = nn.Linear(hidden5, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2, hidden6)
        self.fc2 = nn.Linear(hidden6, hidden6)
        self.fc3 = nn.Linear(hidden6, 2)
        self.fc4 = nn.Linear(2, 2)

        self.t1 = RC1
        self.t2 = RC2
        self.t3 = RC3
        self.t4 = RC4
        self.t5 = RC5

        self.drop1 = nn.Dropout(p=dropout1)
        self.drop2 = nn.Dropout(p=dropout2)
        self.drop3 = nn.Dropout(p=dropout3)

    def forward(self, x):
        y = x
        x = self.relu(self.fc1_1(x))
        out1 = self.relu(self.fc1_2(x))
        x = self.relu(self.fc1_3(out1))
        if self.t1 == 1:
            out2 = self.relu(self.fc1_4(x + out1))
        else:
            out2 = self.relu(self.fc1_4(x))
        x = self.relu(self.fc1_5(out2))
        if self.t2 == 1:
            x = self.fc1_6(x + out2)
        else:
            x = self.fc1_6(x)
        x = self.drop1(x)

        y = self.relu(self.fc2_1(y))
        y1 = self.relu(self.fc2_2(y))
        y = self.relu(self.fc2_3(y1))
        if self.t5 == 1:
            y = self.fc2_4(y + y1)
        else:
            y = self.fc2_4(y)
        y = self.drop2(y)

        out3 = torch.cat([x, y], dim=1)
        out = out3
        out = self.fc1(out)
        out4 = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        if self.t3 == 1:
            out = self.fc3(out + out4)
        else:
            out = self.fc3(out)
        out = self.relu(out)
        if self.t4 == 1:
            out = self.fc4(out + out3)
        else:
            out = self.fc4(out)
        out = self.drop3(out)
        return out
