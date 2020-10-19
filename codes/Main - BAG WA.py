import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
from itertools import combinations

def rmse(y, p):
    return mse(y, p)**0.5

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Main - BAG Pred.csv')
name = ['MLR', 'DTR(6)', 'PR(4)']
r2_v = [0.833, 0.667, 0.5]
comb_names = []
comb_r2 = []
for i in range(1, len(name)+1):
    m = combinations(name, i)
    for j in m:
        comb_names.append(list(j))
for i in range(1, len(r2_v)+1):
    m = combinations(r2_v, i)
    for j in m:
        comb_r2.append(list(j))
mse_f = []
rmse_f = []
mae_f = []
mdae_f = []
evs_f = []
r2_f = []
y = data['Actual']
for i, j in zip(comb_names, comb_r2):
    print(i)
    df = data[i]
    for k, l in zip(i, j):
        df[k] = (l/sum(j))*df[k]
    p = df.sum(axis=1)
    mse_f.append(mse(y, p))
    rmse_f.append(rmse(y, p))
    mae_f.append(mae(y, p))
    mdae_f.append(mdae(y, p))
    evs_f.append(evs(y, p))
    r2_f.append(r2(y, p))
d = {}
d['Combinations'] = comb_names
d['MSE'] = mse_f
d['RMSE'] = rmse_f
d['MAE'] = mae_f
d['MDAE'] = mdae_f
d['EVS'] = evs_f
d['R2'] = r2_f
df = pd.DataFrame(d, columns=['Combinations', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
print(df)
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Main - BAR WA.csv', index=False)