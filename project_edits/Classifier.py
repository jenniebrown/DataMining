'''
Jennifer Brown
CSE 347
'''
import csv 
import sys
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import cross_validation
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
		X = []
		flag = True
		for line in fp:
			if flag:
				flag = False
				continue
			X.append([float(x) for x in line.strip().split(',')])

	X = np.array(X)
	if file_name.find("test") == -1:
		X,y = X[:,:-1],X[:,-1]
		return X, y
	else: 
		return X

def loadTests(file_name):
	with open(file_name, 'r') as fp:
		y = []
		flag = True
		for line in fp:
			if flag:
				flag = False
				continue
			y.append([float(x) for x in line.strip().split(',')])

	y = np.array(y)
	y = y[:,2]
	return y

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

def toFile(X,file_name):
	'''
	write output to file 

	input: 
		X: data to be written
		file_name: name of output file
	'''
	with open(file_name, 'w') as fp:
		fp.write('Id,Failure,Success\n')
		for [ident, fail, succ] in X:
			fp.write('{},{},{}\n'.format(ident,fail,succ))

def crossVal(X, y, clf_name, **args):
	'''
	function to use cross validation to evaluate estimator
	
	input:
		X: numpy array representing features
		y: numpy array representing target
		clf_name: name of classifier 
		args: list of classifier specific arguments

	output: 
		scores of cross-validated classifier
	'''
	CLF = globals()[clf_name]
	clf = CLF(**args)
	scores = cross_validation.cross_val_score(clf, X, y, cv = 10, scoring='log_loss')
	return scores
