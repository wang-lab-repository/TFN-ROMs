import numpy as np
import pandas as pd
import torch
from utils import feature_impute, feature_impute_exiting, rf_fill
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

# path = "data.xlsx"
# drop_features = ['RR']
# need_statistic_filling = ['Pore size (Ǻ)']
# need_model_filling = ['RCA']
# needed_model_filling = ['Pore size (Ǻ)']
# batch_size = 128


def get_data(path, drop_features, need_statistic_filling, need_model_filling, needed_model_filling,
             batch_size, device, zoom_mode=0):
    random = 295
    data = pd.read_excel(path)
    x = data.iloc[:, 0:12]
    y = data.iloc[:, 12:14]
    x.drop(drop_features, axis=1, inplace=True)

    filling = x[need_statistic_filling].median()
    x[need_statistic_filling] = x[need_statistic_filling].fillna(dict(filling), inplace=False)
    model = []
    for target, use_f in zip(need_model_filling, needed_model_filling):
        x, m = feature_impute(x, target, use_f)
        model.append(m)
        rf_fill(x, target)
    # Category characteristics
    # oh: 'Type', 'Shape', 'Phase'
    # ro: 'Charge'
    # 对Charge进行排序
    Charge_ordinal = []
    for i in range(x.shape[0]):
        if x.iloc[i]['Charge'] == '-':
            Charge_ordinal.append(1)
        else:
            Charge_ordinal.append(2)
    x['Charge_ordinal'] = Charge_ordinal
    x.drop(['Charge', 'Type'], axis=1, inplace=True)
    # 将其余的Category characteristics使用oh进行process
    x = pd.get_dummies(x)

    # 这里可以设置很多东西
    if zoom_mode==0:
        std = x.std()
        mean = x.mean()
        x = (x - mean) / std
    elif zoom_mode==1:
        min = x.min()
        max = x.max()
        x = (x - min) / (max - min)
    elif zoom_mode==2:
        min = x.min()
        max = x.max()
        x = (x - min) / max
    elif zoom_mode==3:
        min = x.min()
        max = x.max()
        x = (x - min) / max
    elif zoom_mode==4:
        min = x.min()
        max = x.max()
        x = min + (x-min)/(max-min)*(x-min)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.20, random_state=random)

    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    y_test = torch.from_numpy(y_test.values).float()
    x_train1, y_train1, x_test1, y_test1 = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)
    train_data = Data.TensorDataset(x_train1, y_train1)
    train_load = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    return x_train1, y_train1, x_test1, y_test1, train_load
