from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
import math

path = "../data.xlsx"


def get_need_max_correlation(x, need_model_filling):
    index_dict = {0: 'Size (nm)', 1: 'Pore size (Ǻ)', 2: 'Bond-ing',
                  3: 'Loading', 4: 'RR', 5: 'RCA',
                  6: 'CWP', 7: 'CSP'}
    max = 0
    temp_feature = ''
    cor_search_list = x.corr()[need_model_filling].values.reshape(-1, )
    for index in range(len(cor_search_list)):
        temp = math.fabs(cor_search_list[index])
        if temp == 1 or max > math.fabs(temp):
            continue
        elif max < math.fabs(temp):
            max = math.fabs(temp)
            temp_feature = index_dict[index]
    return temp_feature


def get_null_columns(path):
    data = pd.read_excel(path)
    x = data.iloc[:, 0:12]
    y = data.iloc[:, 12:14]
    x_temp = x.copy()
    columns = x.columns
    null_series = x.isnull().sum()
    drop_features = []
    need_statistic_filling = []
    need_model_filling = []
    needed_model_filling = []
    for c in columns:
        null_rate = null_series[c] / x.shape[0]
        if null_rate == 0:
            continue
        if null_rate > 0.4:
            drop_features.append(c)
        elif null_rate <= 0.05:
            need_statistic_filling.append(c)
        else:
            need_model_filling.append(c)
        # print(c, null_series[c]/x.shape[0])
    needed_model_filling.append(get_need_max_correlation(x_temp, need_model_filling))
    return drop_features, need_statistic_filling, need_model_filling, needed_model_filling


def feature_impute(df_all, target, use_f):
    df_mv = df_all[[use_f, target]].copy()
    df_mv = df_mv.dropna()
    x = np.array(df_mv[use_f]).reshape(-1, 1)
    y = np.array(df_mv[target]).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=666)
    # print("RF: ")
    rf = RandomForestRegressor(random_state=666)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    scores = cross_val_score(rf, x_train, y_train, cv=3, scoring='r2')
    score = np.mean(scores)
    # print("Train R2: %.3f (+/- %.3f) \n" % (score, scores.std()))
    # print("Test R2: %.3f\n" % (r2_score(y_test, pred)))

    # As RF is best:

    name = target + "_imputed"
    p = rf.predict(np.array(df_all[use_f]).reshape(-1, 1))
    df_all[name] = p

    return df_all, rf


def feature_impute_exiting(df_all, target, use_f, model):
    # As RF is best:
    name = target + "_imputed"
    p = model.predict(np.array(df_all[use_f]).reshape(-1, 1))
    df_all[name] = p
    return df_all


def rf_fill(df_all, target):
    df_all[target] = df_all[target].fillna(df_all[target + "_imputed"])
    # drop unrequired featues
    df = df_all.drop([target + "_imputed"], axis=1, inplace=True)
