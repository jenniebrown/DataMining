'''
Jennifer Brown
CSE 347
HW 1
VinoClassifier
'''
import csv 
import sys
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

def load(file_name):
	'''
	load data file into two Numpy arrays: X features, y targets

	input:
		file_name: name of file to be loaded
	output:
		X: numpy array representing features
		y: numpy array representing targets
	'''
	np.set_printoptions(threshold = np.nan)
	with open(file_name, 'r') as fp:
		vino_iter = csv.reader(fp, delimiter = ',', quotechar = '"')
		data = [data for data in vino_iter]

	d = np.asarray(data)
	da = d[1:,:]
	data_array = da.astype(np.float)
	
	X = data_array[:,0:-1]
	y = data_array[:,-1]
	return X, y

def train(X, y, clf_name, **args):
	'''
	train data using given data

	input:
		X: numpy array representing features
		y: numpy array representing target
		clf_name: name of classifier ('DecisionTreeClassifier',
		'LinearSVC')
		args: list of classifier specific arguments, e.g.,
		max_leaf_nodes

	output: classifier: trained classifier
	'''

	CLF = globals()[clf_name]
	clf = CLF(**args)
	return clf.fit(X,y)


