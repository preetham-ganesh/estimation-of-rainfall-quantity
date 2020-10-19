import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Rainfall\\Results\\Final comp.csv')
font = {'family' : 'Times New Roman', 'size'   : 20}
plt.rc('font', **font)
figure(num=None, figsize=(20, 10))
n = list(range(0, 30))
pred1 = list(data['Model 1'])
pred2 = list(data['Model 2'])
pred3 = list(data['Model 3'])
y = list(data['Actual'])
plt.plot(n, y[100:130], '-', color='blue')
plt.plot(n, pred1[100:130], '-', color='black')
plt.plot(n, pred2[100:130], '-', color='yellow')
plt.plot(n, pred3[100:130], '-', color='red')
plt.xlabel('Data Points')
plt.ylabel('Rainfall')
plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
plt.legend(['True', 'PR[4]', 'BAR[PR[4]]', 'Stacking'], loc='upper left')
plt.show()