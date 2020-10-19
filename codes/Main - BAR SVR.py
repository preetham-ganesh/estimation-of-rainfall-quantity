import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
import numpy as np

def rmse(y_t, y_p):
    return (mse(y_t, y_p))**0.5

rkf = RepeatedKFold(n_splits=10, n_repeats=10)
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
data = data.drop(columns=['Index', 'District'])
mse_t = []
rmse_t = []
mae_t = []
mdae_t = []
evs_t = []
r2_t = []
c = 0
for tr_i, ts_i in rkf.split(data):
    print(c)
    train, test = data.iloc[tr_i], data.iloc[ts_i]
    train_x = train.drop(columns=['Rainfall'])
    train_y = train['Rainfall']
    test_x = test.drop(columns=['Rainfall'])
    test_y = test['Rainfall']
    model = BaggingRegressor(n_estimators=50, base_estimator=SVR(kernel='linear'))
    model.fit(train_x, train_y)
    ts_p = model.predict(test_x)
    mse_t.append(mse(test_y, ts_p))
    rmse_t.append(rmse(test_y, ts_p))
    mae_t.append(mae(test_y, ts_p))
    mdae_t.append(mdae(test_y, ts_p))
    evs_t.append(evs(test_y, ts_p))
    r2_t.append(r2(test_y, ts_p))
    c += 1
print(np.mean(mse_t))
print(np.mean(rmse_t))
print(np.mean(mae_t))
print(np.mean(mdae_t))
print(np.mean(evs_t))
print(np.mean(r2_t))