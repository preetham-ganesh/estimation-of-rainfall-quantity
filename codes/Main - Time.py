import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
import numpy as np

from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=10, n_repeats=10)
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
data = data.drop(columns=['Index', 'District'])
models = [RandomForestRegressor(n_estimators=100, max_depth=9),
          ExtraTreesRegressor(n_estimators=90, max_depth=11),
          GradientBoostingRegressor(n_estimators=50, max_depth=5),
          XGBRegressor(n_estimators=50, max_depth=5)]
names = ['RFR', 'ETR', 'GBR', 'XGBR']
tr_f = []
ts_r_f = []
ts_s_f = []
for i, j in zip(models, names):
    tr_p = []
    ts_r_p = []
    ts_s_p = []
    c = 0
    for tr_i, ts_i in rkf.split(data):
        print(j, c)
        train, test = data.iloc[tr_i], data.iloc[ts_i]
        train_x = train.drop(columns=['Rainfall'])
        train_y = train['Rainfall']
        test_x = test.drop(columns=['Rainfall'])
        model = i
        t1 = time.time()
        model.fit(train_x, train_y)
        t2 = time.time()
        model.predict(train_x)
        t3 = time.time()
        model.predict(test_x)
        t4 = time.time()
        tr_p.append(t2-t1)
        ts_r_p.append(t3-t2)
        ts_s_p.append(t4-t3)
        c += 1
    tr_f.append(np.mean(tr_p))
    ts_r_f.append(np.mean(ts_r_p))
    ts_s_f.append(np.mean(ts_s_p))
d = {}
d['Training'] = tr_f
d['Testing Train'] = ts_r_f
d['Testing Test'] = ts_s_f
df = pd.DataFrame(d, columns=['Training', 'Testing Train', 'Testing Test'])
print(df)
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Time.csv', index=False)