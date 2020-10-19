import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
import numpy as np

def rmse(y, p):
    return mse(y, p)**0.5

rkf = RepeatedKFold(n_splits=10, n_repeats=10)
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
data = data.drop(columns=['Index', 'District'])
est = list(range(1, 11))
nest = []
for i in est:
    nest.append(i*10)
print(nest)
mse_tr_f = []
mse_ts_f = []
rmse_tr_f = []
rmse_ts_f = []
mae_tr_f = []
mae_ts_f = []
mdae_tr_f = []
mdae_ts_f = []
evs_tr_f = []
evs_ts_f = []
r2_tr_f = []
r2_ts_f = []
for i in nest:
    c = 0
    mse_tr_t = []
    mse_ts_t = []
    rmse_tr_t = []
    rmse_ts_t = []
    mae_tr_t = []
    mae_ts_t = []
    mdae_tr_t = []
    mdae_ts_t = []
    evs_tr_t = []
    evs_ts_t = []
    r2_tr_t = []
    r2_ts_t = []
    for tr_i, ts_i in rkf.split(data):
        print(i, c)
        c += 1
        train, test = data.iloc[tr_i], data.iloc[ts_i]
        train_x = train.drop(columns=['Rainfall'])
        train_y = train['Rainfall']
        test_x = test.drop(columns=['Rainfall'])
        test_y = test['Rainfall']
        model = AdaBoostRegressor(n_estimators=i)
        model.fit(train_x, train_y)
        tr_p = model.predict(train_x)
        ts_p = model.predict(test_x)
        mse_tr_t.append(mse(train_y, tr_p))
        mse_ts_t.append(mse(test_y, ts_p))
        rmse_tr_t.append(rmse(train_y, tr_p))
        rmse_ts_t.append(rmse(test_y, ts_p))
        mae_tr_t.append(mae(train_y, tr_p))
        mae_ts_t.append(mae(test_y, ts_p))
        mdae_tr_t.append(mdae(train_y, tr_p))
        mdae_ts_t.append(mdae(test_y, ts_p))
        evs_tr_t.append(evs(train_y, tr_p))
        evs_ts_t.append(evs(test_y, ts_p))
        r2_tr_t.append(r2(train_y, tr_p))
        r2_ts_t.append(r2(test_y, ts_p))
    mse_tr_f.append(np.mean(mse_tr_t))
    mse_ts_f.append(np.mean(mse_ts_t))
    rmse_tr_f.append(np.mean(rmse_tr_t))
    rmse_ts_f.append(np.mean(rmse_ts_t))
    mae_tr_f.append(np.mean(mae_tr_t))
    mae_ts_f.append(np.mean(mae_ts_t))
    mdae_tr_f.append(np.mean(mdae_tr_t))
    mdae_ts_f.append(np.mean(mdae_ts_t))
    evs_tr_f.append(np.mean(evs_tr_t))
    evs_ts_f.append(np.mean(evs_ts_t))
    r2_tr_f.append(np.mean(r2_tr_t))
    r2_ts_f.append(np.mean(r2_ts_t))
d = {}
d['No. of Estimators'] = nest
d['Train MSE'] = mse_tr_f
d['Test MSE'] = mse_ts_f
d['Train RMSE'] = rmse_tr_f
d['Test RMSE'] = rmse_ts_f
d['Train MAE'] = mae_tr_f
d['Test MAE'] = mae_ts_f
d['Train MDAE'] = mdae_tr_f
d['Test MDAE'] = mdae_ts_f
d['Train EVS'] = evs_tr_f
d['Test EVS'] = evs_ts_f
d['Train R2'] = r2_tr_f
d['Test R2'] = r2_ts_f
df = pd.DataFrame(d, columns=['No. of Estimators', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'Train MDAE', 'Test MDAE', 'Train EVS', 'Test EVS', 'Train R2', 'Test R2'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Adaboost\\Estimators.csv', index=False)