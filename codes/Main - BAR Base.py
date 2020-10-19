import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=10, n_repeats=10)
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
data = data.drop(columns=['Index', 'District'])
base = [lm.LinearRegression(), DecisionTreeRegressor(max_depth=6), lm.LinearRegression()]
name = ['MLR', 'DTR(6)', 'PR(4)']
df = pd.DataFrame()
c = 0
for tr_i, ts_i in rkf.split(data):
    train, test = data.iloc[tr_i], data.iloc[ts_i]
    train_x = train.drop(columns=['Rainfall'])
    train_y = train['Rainfall']
    test_x = test.drop(columns=['Rainfall'])
    test_y = test['Rainfall']
    d = {}
    for i, j in zip(base, name):
        print(j, c)
        if j == 'PR(4)':
            poly = pf(degree=4)
            train_x = poly.fit_transform(train_x)
            test_x = poly.fit_transform(test_x)
        model = BaggingRegressor(n_estimators=50, base_estimator=i)
        model.fit(train_x, train_y)
        ts_p = model.predict(test_x)
        d[j] = list(ts_p)
    d['Actual'] = list(test_y)
    df1 = pd.DataFrame(d, columns=['None', 'MLR', 'DTR(6)', 'PR(4)', 'Actual'])
    df = df.append(df1)
    c += 1
print(df)