import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error as mae
from sklearn.model_selection import RepeatedKFold
import numpy as np

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
rkf = RepeatedKFold(n_splits=10, n_repeats=10)
models = [DecisionTreeRegressor(max_depth=6)]#, LinearRegression()]
names = ['DTR(6)']#, 'PR(4)']
nest = [10, 100, 200, 300, 400, 500]
data = data.drop(columns=['Index', 'District'])
d = {}
for i, j in zip(models, names):
    mae_f = []
    for k in nest:
        mae_t = []
        c = 0
        for tr_i, ts_i in rkf.split(data):
            print(j, k, c)
            train, test = data.iloc[tr_i], data.iloc[ts_i]
            train_x = train.drop(columns=['Rainfall'])
            train_y = train['Rainfall']
            test_x = test.drop(columns=['Rainfall'])
            test_y = test['Rainfall']
            model = AdaBoostRegressor(base_estimator=i, n_estimators=k)
            model.fit(train_x, train_y)
            ts_p = model.predict(test_x)
            mae_t.append(mae(test_y, ts_p))
            c += 1
        mae_f.append(np.mean(mae_t))
    d[j] = mae_f
d['Nest'] = nest
df = pd.DataFrame(d, columns=['Nest', 'MLR'])
print(df)