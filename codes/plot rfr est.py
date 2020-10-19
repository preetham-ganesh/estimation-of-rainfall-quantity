import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
#from scipy.interpolate import interp1d
from matplotlib.pyplot import figure

font = {'family' : 'Times New Roman',
        'size'   : 28}

plt.rc('font', **font)
figure(num=None, figsize=(17, 5))
data = pd.read_csv('C:\\Users\\Preetham G\\Downloads\\Main - RFR Est1.csv')
plt.plot(data['Estimators'], data['MSE'], 'r*--', label='MSE')
plt.xlabel('Estimators')
plt.ylabel('Values')
plt.legend(loc='bottom right')
plt.xticks(data['Estimators'])
plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
plt.show()