import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model as lm

train = pd.read_csv('C:\\Users\\Preetham G\\Downloads\\train.csv')
test = pd.read_csv('C:\\Users\\Preetham G\\Downloads\\test.csv')

stack_train = pd.read_csv('C:\\Users\\Preetham G\\Downloads\\train_kfold.csv')
stack_test = pd.read_csv('C:\\Users\\Preetham G\\Downloads\\predict_test.csv')

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\All District Combined.csv')

poly = pf(degree=4)
train_x = train.drop(columns=['District', 'Index', 'Rainfall'])
test_x = test.drop(columns=['District', 'Index', 'Rainfall'])
train_y = train['Rainfall']
test_y = test['Rainfall']

stack_train_x = stack_train[['ETR', 'BAR']]
stack_test_x = stack_test[['ETR', 'BAR']]
stack_train_y = stack_train['Actual']
stack_test_y = stack_test['Actual']

poly = pf(degree=4)
poly_tr_x = poly.fit_transform(train_x)
poly_ts_x = poly.fit_transform(test_x)
poly_st_tr_x = poly.fit_transform(stack_train_x)
poly_st_ts_x = poly.fit_transform(stack_test_x)

model1 = lm.LinearRegression()
model2 = BaggingRegressor(n_estimators=50, base_estimator=lm.LinearRegression())
model3 = lm.LinearRegression()

model1.fit(poly_tr_x, train_y)
model2.fit(poly_tr_x, train_y)
model3.fit(poly_st_tr_x, stack_train_y)

pred1 = model1.predict(poly_ts_x)
pred2 = model2.predict(poly_ts_x)
pred3 = model3.predict(poly_st_ts_x)

d = {}
maxi = max(list(data['Rainfall']))
pred1_c = [i * maxi for i in pred1]
pred2_c = [i * maxi for i in pred2]
pred3_c = [i * maxi for i in pred3]
y_c = [i * maxi for i in test_y]
d['Model 1'] = list(pred1_c)
d['Model 2'] = list(pred2_c)
d['Model 3'] = list(pred3_c)
d['Actual'] = list(y_c)
df = pd.DataFrame(d, columns=['Model 1', 'Model 2', 'Model 3', 'Actual'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Final comp.csv', index=False)
"""
font = {'family' : 'Times New Roman', 'size'   : 20}
plt.rc('font', **font)
figure(num=None, figsize=(20, 8))

pred1 = list(pred1)
pred2 = list(pred2)
pred3 = list(pred3)




    
n = list(range(0, 100))
plt.plot(n, y_c[0:100], 'b-')
plt.plot(n, pred1_c[0:100], 'r-')
plt.plot(n, pred2_c[0:100], 'g-')
plt.plot(n, pred3_c[0:100], 'y-')
plt.xlabel('Data Points')
plt.ylabel('Rainfall')
plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
plt.legend(['True', 'Model1', 'Model2', 'Model3'], loc='upper left')
plt.show()"""