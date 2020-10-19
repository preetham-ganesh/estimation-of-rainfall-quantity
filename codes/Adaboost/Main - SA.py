import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=10, n_repeats=10)
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
models = [None, LinearRegression(), DecisionTreeRegressor(max_depth=6), SVR(kernel='linear'), LinearRegression()]
names = ['None', 'MLR', 'DTR(6)', 'SVR(L)', 'PR(4)']
data = data.drop(columns=['Index', 'District'])
poly = PolynomialFeatures(degree=4)
c = 0
for tr_i, ts_i in rkf.split(data):
    train, test = data.iloc[tr_i], data.iloc[ts_i]
    train_x = train.drop(columns=['Rainfall'])
    train_y = train['Rainfall']
    test_x = test.drop(columns=['Rainfall'])
    test_y = test['Rainfall']
    d = {}
    for i, j in zip(models, names):
        model = AdaBoostRegressor(base_estimator=i)
        if j == 'PR(4)':
            train_x = poly.fit_transform(train_x)
            test_x = poly.fit_transform(test_x)
        model.fit(train_x, train_y)
        ts_p = model.predict(test_x)
        d[j] = list(ts_p)
        print(j, c)
    c += 1
    d['True'] = list(test_y)
    df = pd.DataFrame(d, columns=['None', 'MLR', 'DTR(6)', 'PR(4)', 'SVR(L)', 'True'])
    df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Adaboost\\Main Results\\R' + str(c) + '.csv', index=False)
    print('R' + str(c) + ' Generated')
    print()