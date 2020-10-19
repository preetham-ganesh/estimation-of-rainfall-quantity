import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
from itertools import combinations
import numpy as np

def rmse(y, p):
    return mse(y, p)**0.5

names = ['None', 'MLR', 'DTR(6)', 'SVR(L)', 'PR(4)']
comb_names = []
for i in range(1, len(names)+1):
    m = combinations(names, i)
    for j in m:
        comb_names.append(list(j))
mse_f = []
rmse_f = []
mae_f = []
mdae_f = []
evs_f = []
r2_f = []
for i in comb_names:
    mse_t = []
    rmse_t = []
    mae_t = []
    mdae_t = []
    evs_t = []
    r2_t = []
    for c in range(1, 101):
        print(i, c)
        data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Adaboost\\Main Results\\R' + str(c) + '.csv')
        y = data['True']
        data = data[i]
        p = data.mean(axis=1)
        mse_t.append(mse(y, p))
        rmse_t.append(rmse(y, p))
        mae_t.append(mae(y, p))
        mdae_t.append(mdae(y, p))
        evs_t.append(evs(y, p))
        r2_t.append(r2(y, p))
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
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Adaboost\\Main - Simple Average.csv', index=False)