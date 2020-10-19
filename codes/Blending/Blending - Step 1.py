import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv('C:\\Users\\Preetham G\\Downloads\\train.csv')
data = data.drop(columns=['Index', 'District'])
base = [RandomForestRegressor(n_estimators=100, max_depth=10), ExtraTreesRegressor(n_estimators=90, max_depth=15), GradientBoostingRegressor(n_estimators=60, max_depth=5), XGBRegressor(n_estimators=50, max_depth=5), BaggingRegressor(n_estimators=50, base_estimator=lm.LinearRegression())]
name = ['RFR', 'ETR', 'GBR', 'XGBR', 'BAR']
df1 = pd.DataFrame()
c = 0
train, test = tts(data, test_size=0.1)
train_x = train.drop(columns=['Rainfall'])
train_y = train['Rainfall']
test_x = test.drop(columns=['Rainfall'])
test_y = test['Rainfall']
d1 = {}
for i, j in zip(base, name):
    print(j, c)
    if j == 'BAR':
        poly = pf(degree=4)
        train_x = poly.fit_transform(train_x)
        test_x = poly.fit_transform(test_x)
    model = i
    model.fit(train_x, train_y)
    ts_p = model.predict(test_x)
    d1[j] = list(ts_p)
d1['Actual'] = list(test_y)
df_ts = pd.DataFrame(d1, columns=['RFR', 'ETR', 'GBR', 'XGBR', 'BAR', 'Actual'])
c += 1
df_ts.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Blend - Step 1.csv', index=False)