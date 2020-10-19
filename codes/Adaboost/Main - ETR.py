import pandas as pd
from sklearn.tree import ExtraTreeRegressor
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
dep = [2, 3, 4, 5, 6, 7]
saml = [1, 2, 3, 4, 5, 6]
sams = [2, 3, 4, 5, 6, 7]
dep_f = []
saml_f = []
sams_f = []
mse_f = []
rmse_f = []
mae_f = []
mdae_f = []
evs_f = []
r2_f = []
for i in dep:
    for j in saml:
        for k in sams:
            c = 0
            mse_t = []
            rmse_t = []
            mae_t = []
            mdae_t = []
            evs_t = []
            r2_t = []
            for tr_i, ts_i in rkf.split(data):
                print(i, j, k, c)
                train, test = data.iloc[tr_i], data.iloc[ts_i]
                train_x = train.drop(columns=['Rainfall'])
                train_y = train['Rainfall']
                test_x = test.drop(columns=['Rainfall'])
                test_y = test['Rainfall']
                model = ExtraTreeRegressor(criterion='mse', splitter='best', max_depth=i, min_samples_leaf=j, min_samples_split=k)
                model.fit(train_x, train_y)
                ts_p = model.predict(test_x)
                mse_t.append(mse(test_y, ts_p))
                rmse_t.append(rmse(test_y, ts_p))
                mae_t.append(mae(test_y, ts_p))
                mdae_t.append(mdae(test_y, ts_p))
                evs_t.append(evs(test_y, ts_p))
                r2_t.append(r2(test_y, ts_p))
                c += 1
                dep_f.append(i)
                saml_f.append(j)
                sams_f.append(k)
                mse_f.append(np.mean(mse_t))
                rmse_f.append(np.mean(rmse_t))
                mae_f.append(np.mean(mae_t))
                mdae_f.append(np.mean(mdae_t))
                evs_f.append(np.mean(evs_t))
                r2_f.append(np.mean(r2_t))
d = {}
d['Max Depth'] = dep_f
d['Min Samples Leaf'] = saml_f
d['Min Samples Split'] = sams_f
d['MSE'] = mse_f
d['RMSE'] = rmse_f
d['MAE'] = mae_f
d['MDAE'] = mdae_f
d['EVS'] = evs_f
d['R2'] = r2_f
df = pd.DataFrame(d, columns=['Max Depth', 'Min Samples Leaf', 'Min Samples Leaf', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Main - ETR.csv', index=False)