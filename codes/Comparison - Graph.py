import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=10, n_repeats=10)

data1 = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\All District Combined.csv')
maxi = max(list(data1['Rainfall']))
mini = min(list(data1['Rainfall']))

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
poly = pf(degree=4)
train, test = tts(data, test_size=0.1)
train_x = train.drop(columns=['Rainfall', 'District'])
train_y = train['Rainfall']
test_x = test.drop(columns=['Rainfall', 'District'])
test_y = test['Rainfall']
tr_x = poly.fit_transform(train_x)
ts_x = poly.fit_transform(test_x)
model1 = lm.LinearRegression()
model1.fit(tr_x, train_y)
pr_f = model1.predict(ts_x)
print('Success')
model2_m = [ExtraTreesRegressor(n_estimators=90, max_depth=11), BaggingRegressor(n_estimators=70, base_estimator=lm.LinearRegression())]
model2_n = ['ETR', 'BAR']
df_tr = pd.DataFrame()
for tr_i, ts_i in rkf.split(train):
    d_tr, d_ts = train.iloc[tr_i], train.iloc[ts_i]
    d_tr_x = d_tr.drop(columns=['Rainfall', 'District'])
    d_tr_y = d_tr['Rainfall']
    d_ts_x = d_ts.drop(columns=['Rainfall', 'District'])
    d_ts_y = d_ts['Rainfall']
    d = {}
    c = 0
    for i, j in zip(model2_m, model2_n):
        print(j, c)
        if j == 'BAR':
            d_tr_x = poly.fit_transform(d_tr_x)
            d_ts_x = poly.fit_transform(d_ts_x)
        model = i
        model.fit(d_tr_x, d_tr_y)
        ts_p = model.predict(d_ts_x)
        d[j] = list(ts_p)
    d['Actual'] = d_ts_y
    df = pd.DataFrame(d, columns=['ETR', 'BAR', 'Actual'])
    df_tr = df_tr.append(df)
df_ts = pd.DataFrame()
d = {}
for i, j in zip(model2_m, model2_n):
    print(j)
    model = i
    if j == 'BAR':
        model.fit(tr_x, train_y)
        ts_p = model.predict(ts_x)
    else:
        model.fit(train_x, train_y)
        ts_p = model.predict(test_x)
    d[j] = list(ts_p)
d['Actual'] = list(test_y)
df_ts = pd.DataFrame(d, columns=['ETR', 'BAR', 'Actual'])
df_tr_x = df_tr.drop(columns=['Actual'])
df_tr_y = df_tr['Actual']
df_ts_x = df_ts.drop(columns=['Actual'])
df_ts_y = df_ts['Actual']
df_tr_x_p = poly.fit_transform(df_tr_x)
df_ts_x_p = poly.fit_transform(df_ts_x)
model2 = lm.LinearRegression()
model2.fit(df_tr_x_p, df_tr_y)
sta_f = model2.predict(df_ts_x)

from matplotlib.pyplot import figure

font = {'family' : 'Times New Roman', 'size'   : 20}

plt.rc('font', **font)
figure(num=None, figsize=(20, 8))
y = list(test['Rainfall'])
pr_f = list(pr_f)
y_c = []
for i in y:
    y_c.append(i*maxi)
pr_c = []
for i in pr_f:
    pr_c.append(i*maxi)
sta_c = []
for i in sta_f:
    sta_c.append(i*maxi)
n = list(range(0, 100))
plt.plot(n, y_c[0:100], 'b-')
plt.plot(n, pr_c[0:100], 'r-')
plt.plot(n, sta_c[0:100], 'g-')
plt.xlabel('Data Points')
plt.ylabel('Rainfall')
plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
plt.legend(['True', 'Model1', 'Model2'], loc='upper left')
plt.show()