import pandas as pd
from sklearn import linear_model as lm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as evs
from itertools import combinations

train = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Blend - Train.csv')
test = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Blend - Test.csv')

name = list(train.columns.values)
name.remove('Actual')
comb_names = []
for i in range(1, len(name)+1):
    m = combinations(name, i)
    for j in m:
        comb_names.append(list(j))
        
models = [lm.LinearRegression(), DecisionTreeRegressor(max_depth=6), lm.LinearRegression(), SVR(kernel='linear', epsilon=0.1, C=1)]
model_names = ['MLR', 'DTR[6]', 'PR[4]', 'SVR[L]']
d = {}
for j, k in zip(models, model_names):
    mse_t = []
    evs_t = []
    for i in comb_names:
        print(k, i)
        train_x = train[i]
        train_y = train['Actual']
        test_x = test[i]
        test_y = test['Actual']    
        if k == 'PR[4]':
            poly = pf(degree=4)
            train_x = poly.fit_transform(train_x)
            test_x = poly.fit_transform(test_x)
        model = j
        model.fit(train_x, train_y)
        ts_p = model.predict(test_x)
        mse_t.append(mse(test_y, ts_p))
        evs_t.append(evs(test_y, ts_p))
    l = 'MSE - ' + k
    m = 'EVS - ' + k
    d[l] = mse_t
    d[m] = evs_t
d['Combination'] = comb_names
df = pd.DataFrame(d, columns=['Combination', 'MSE - MLR', 'MSE - PR[4]', 'MSE - DTR[6]', 'MSE - SVR[L]', 'EVS - MLR', 'EVS - PR[4]', 'EVS - DTR[6]', 'EVS - SVR[L]'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Blending - Final.csv', index=False)