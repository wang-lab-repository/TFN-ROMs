from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import warnings
from loss import MyLoss
from model import Mymodel
from get_data import get_data
from utils import get_null_columns

warnings.filterwarnings("ignore")
epoch = 1000
lr = 0.001687534952708282
lr_min = 1.953914135289555e-05
# optimizer_name
batch_size = 128
path = "../data.xlsx"
hidden1 = 70
hidden2 = 155
hidden3 = 85
hidden4 = 175
hidden5 = 20
hidden6 = 95
RC1 = 0
RC2 = 0
RC3 = 1
RC4 = 1
RC5 = 0
dropout1 = 0.05
dropout2 = 0.25
dropout3 = 0.0
device = 'cpu'
partition = 9.0
is_train = False
drop_features, need_statistic_filling, need_model_filling, needed_model_filling = get_null_columns(path)
# print(drop_features, need_statistic_filling, need_model_filling, needed_model_filling)
x_train1, y_train1, x_test1, y_test1, train_load = get_data(path, drop_features, need_statistic_filling,
                                                            need_model_filling, needed_model_filling,
                                                            batch_size=batch_size, device=device)


def train():
    net = Mymodel(hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, hidden4=hidden4, hidden5=hidden5, hidden6=hidden6,
                  RC1=RC1, RC2=RC2, RC3=RC3, RC4=RC4, RC5=RC5,
                  dropout1=dropout1, dropout2=dropout2, dropout3=dropout3)
    if (is_train) == False:
        net.load_state_dict(torch.load('checkpoint/0.715_0.967_last_model.ckpt'), False)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=lr_min)
    loss_func = MyLoss(partition=partition)
    if is_train:
        for i in range(epoch):
            for step, (train_x, train_y) in enumerate(train_load):
                train_pre = net(train_x)
                train_loss = loss_func(train_pre, train_y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
    # Model freeze
    net.eval()
    # Compute R² on the training set
    prediction_train = net(x_train1)
    SSres_train = torch.mean((y_train1 - prediction_train) ** 2)
    SStot_train = torch.mean((y_train1 - torch.mean(y_train1)) ** 2)
    R2_train = 1 - SSres_train / SStot_train
    # Calculate R² on the test set
    prediction_test = net(x_test1)
    # Calculate R² for each output feature on the training set
    R2_train_1 = 1 - torch.mean((y_train1[:, 0] - prediction_train[:, 0]) ** 2) / torch.mean(
        (y_train1[:, 0] - torch.mean(y_train1[:, 0])) ** 2)
    R2_train_2 = 1 - torch.mean((y_train1[:, 1] - prediction_train[:, 1]) ** 2) / torch.mean(
        (y_train1[:, 1] - torch.mean(y_train1[:, 1])) ** 2)
    # Calculate R² for each output feature on the test set
    R2_test_1 = 1 - torch.mean((y_test1[:, 0] - prediction_test[:, 0]) ** 2) / torch.mean(
        (y_test1[:, 0] - torch.mean(y_test1[:, 0])) ** 2)
    R2_test_2 = 1 - torch.mean((y_test1[:, 1] - prediction_test[:, 1]) ** 2) / torch.mean(
        (y_test1[:, 1] - torch.mean(y_test1[:, 1])) ** 2)
    RMSE_train_1=np.sqrt(mean_squared_error(y_train1.detach().numpy()[:, 0],prediction_train.detach().numpy()[:, 0]))
    RMSE_train_2=np.sqrt(mean_squared_error(y_train1.detach().numpy()[:, 1],prediction_train.detach().numpy()[:, 1]))
    RMSE_test_1=np.sqrt(mean_squared_error(y_test1.detach().numpy()[:, 0],prediction_test.detach().numpy()[:, 0]))
    RMSE_test_2=np.sqrt(mean_squared_error(y_test1.detach().numpy()[:, 1],prediction_test.detach().numpy()[:, 1]))
    # Output the final model evaluation results
    print("------------------------R2: Result------------------------")
    print('train: RWP：{:.3f},RSP {:.3f}\n'.format(R2_train_1, R2_train_2))
    print('test: RWP：{:.3f},RSP {:.3f}\n'.format(R2_test_1, R2_test_2))
    print("------------------------RMSE: Result------------------------")
    print('train: RWP：{:.3f},RSP {:.3f}\n'.format(RMSE_train_1, RMSE_train_2))
    print('test: RWP：{:.3f},RSP {:.3f}\n'.format(RMSE_test_1, RMSE_test_2))
