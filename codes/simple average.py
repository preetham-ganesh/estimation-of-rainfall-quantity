import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import RepeatedKFold
import numpy as np
from itertools import combinations

def rmse(y_t, y_p):
    return (mse(y_t, y_p))**0.5

rkf = RepeatedKFold(n_splits=10, n_repeats=10)
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
models = [LinearRegression(), DecisionTreeRegressor(max_depth=6), LinearRegression(), SVR(kernel='linear')]
names = ['MLR', 'DTR(6)', 'PR(4)', 'SVR(L)']
comb_models = []
comb_names = []
for i in range(1, len(models)+1):
    l = combinations(models, i)
    m = combinations(names, i)
    for j in l:
        comb_models.append(list(j))
    for j in m:
        comb_names.append(list(j))
data = data.drop(columns=['Index', 'District'])
mse_f = []
rmse_f = []
mae_f = []
mdae_f = []
evs_f = []
r2_f = []
poly = PolynomialFeatures(degree=4)
for i, j in zip(comb_models, comb_names):
    c = 0
    mse_t = []
    rmse_t = []
    mae_t = []
    mdae_t = []
    evs_t = []
    r2_t = []
    for tr_i, ts_i in rkf.split(data):
        train, test = data.iloc[tr_i], data.iloc[ts_i]
        train_x = train.drop(columns=['Rainfall'])
        train_y = train['Rainfall']
        test_x = test.drop(columns=['Rainfall'])
        test_y = test['Rainfall']
        d = {}
        for k, l in zip(i, j):
            print(j, l, c)
            model = k
            if l == 'PR(4)':
                train_x = poly.fit_transform(train_x)
                test_x = poly.fit_transform(test_x)
            model.fit(train_x, train_y)
            ts_p = model.predict(test_x)
            d[l] = list(ts_p)
        c += 1
        df = pd.DataFrame(d, columns=names)
        ts_p_m = df.mean(axis=1)
        mse_t.append(mse(test_y, ts_p_m))
        rmse_t.append(rmse(test_y, ts_p_m))
        mae_t.append(mae(test_y, ts_p_m))
        mdae_t.append(mdae(test_y, ts_p_m))
        evs_t.append(evs(test_y, ts_p_m))
        r2_t.append(r2(test_y, ts_p_m))
    mse_f.append(np.mean(mse_t))
    rmse_f.append(np.mean(rmse_t))
    mae_f.append(np.mean(mae_t))
    mdae_f.append(np.mean(mdae_t))
    evs_f.append(np.mean(evs_t))
    r2_f.append(np.mean(r2_t))
d = {}
d['Combinations'] = comb_names
d['MSE'] = mse_f
d['RMSE'] = rmse_f
d['MAE'] = mae_f
d['MDAE'] = mdae_f
d['EVS'] = evs_f
d['R2'] = r2_f
df = pd.DataFrame(d, columns=['Combinations', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Simple Average.csv', index=False)