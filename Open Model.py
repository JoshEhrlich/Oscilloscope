import os
from sklearn import datasets
from sklearn import svm    			
import numpy as np
import scipy as sy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn import metrics
import numpy.fft as fftt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from scipy.signal import butter,filtfilt
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import glob
import numpy as np
import pandas as pd
import scipy
from mlxtend.plotting import plot_decision_regions
from scipy import signal
import pickle

filename = "/Users/JoshEhrlich/OneDrive - Queen's University/School/University/MSc/PicoscopeAnalysis/ModelPicoscope/cauteryModelSVM.sav"
loaded_module = pickle.load(open(filename, "rb")

