from sklearn import datasets
from sklearn import svm    			
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

X = np.load("/Users/JoshEhrlich/OneDrive - Queen's University/School/University/MSc/PicoscopeAnalysis/OldWork/x.npy")
#Y = np.load("/Users/JoshEhrlich/OneDrive - Queen's University/School/University/Graduate School/Pico Export Data/sample_data/y.npy")

sample1 = X[2]
print(X.shape)
print(Y.shape)
time1 = sample1[1]
volt1 = sample1[0]


sample1 = X[300]
time2 = sample1[1]
volt2 = sample1[0]

# plt.plot(time1, volt1)
# plt.show()

# plt.plot(time2, volt2)
# plt.show()

X_new = np.array([])
Y_new = np.array([])