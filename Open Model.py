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

'''

This is where the manipulation of the waves takes place

'''

# find average "density" of waves
for i in range(1000):
    sample_i = X[i]
    volt_avg_i = np.mean(sample_i[0])
    X_new = np.append(X_new, [volt_avg_i], axis=0)

# find peaks/crests and then find frequency

avg_frequency = np.array([])
for i in range(1000):
    sample_i = X[i]
    time_i = np.array([])
    for j in range(100):
        if ((sample_i[0][j] > sample_i[0][j - 1]) and (sample_i[0][j] > sample_i[0][j + 1])):
            time_i = np.append(time_i, [sample_i[1][j]], axis=0)

    length = len(time_i)
    frequency = 100 / length
    avg_frequency = np.append(avg_frequency, [frequency])
