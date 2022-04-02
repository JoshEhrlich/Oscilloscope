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

# amount changed between each point#

pts_difference_avg = np.array([])
for i in range(1000):
    sample_i = X[i]
    pts_difference = np.array([])
    for j in range(99):
        pts_difference = np.append(pts_difference, [abs(sample_i[0][j] - sample_i[0][j + 1])], axis=0)

    sample_average = np.mean(pts_difference)
    pts_difference_avg = np.append(pts_difference_avg, [sample_average])


'''

Here I did some brutal building of matrices because I couldnt figure out in-house functions to do that

'''

X_new = np.transpose(X_new)
avg_frequency = np.transpose(avg_frequency)

#Building the classifier array
markers0 = np.full((250),0)
markers1 = np.full((250),1)
markers2 = np.full((250),2)
markers3 = np.full((250),3)

markers = np.array(markers0)
markers = np.append(markers, markers1)
markers = np.append(markers, markers2)
markers = np.append(markers, markers3)
markers = np.transpose(markers)

#Re-formatting arrays to match model requirements
X_new = np.expand_dims(X_new, axis = 0)
X_new = np.transpose(X_new)
avg_frequency = np.expand_dims(avg_frequency, axis = 0)
avg_frequency = np.transpose(avg_frequency)

X_new_and_avg_frequency = np.array([X_new, avg_frequency])
X_new_and_avg_frequency = np.squeeze(X_new_and_avg_frequency)
X_new_and_avg_frequency = np.transpose(X_new_and_avg_frequency)


pts_difference_avg = np.expand_dims(pts_difference_avg, axis = 0)
pts_difference_avg = np.transpose(pts_difference_avg)

X_new_and_pts_diff_avg = np.array([X_new, pts_difference_avg])
X_new_and_pts_diff_avg = np.squeeze(X_new_and_pts_diff_avg)
X_new_and_pts_diff_avg = np.transpose(X_new_and_pts_diff_avg)

'''

This is where I actually performed the SVM. I used the stock code from the website we looked at.

'''

'''
SVM#1 X_new and avg_frequency
'''

C = 1.0
svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_new_and_avg_frequency, markers)
lin_svc = svm.LinearSVC(C=1.0).fit(X_new_and_avg_frequency, markers)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_new_and_avg_frequency, markers)
poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_new_and_avg_frequency, markers)


h = .02  # step size in the mesh

# create a mesh to plot in

x_min, x_max = X_new.min() - 1, X_new.max() + 1
y_min, y_max = avg_frequency.min() - 1, avg_frequency.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
# title for the plots
titles = ['SVC with linear kernel',
       'LinearSVC (linear kernel)',
        'SVC with RBF kernel',
        'SVC with polynomial (degree 3) kernel']
