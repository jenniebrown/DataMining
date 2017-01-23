'''
Jennifer Brown
CSE 347
HW1
Imputer.py
'''

from sklearn.linear_model import LinearRegression
import numpy as NP

def impute(X):
	'''
	impute missing values in last column of X using linear regression
	input: X matrix with missing values
	output: modified X with missing values imputed
	'''
	m = len(X)
	n = len(X[0])
	X_train = NP.array([])
	rowCount = 0
	for r in range(0,m):
		if not NP.isnan(X[r][-1]):
			X_train =NP.concatenate((X_train,X[r]),axis=0)
			rowCount += 1
	X_train = NP.reshape(X_train,(rowCount, n))
	mdl = LinearRegression()
	mdl.fit(X_train[:,0:-2],X_train[:,-1])
	X_predict = X[:,0:-2]
	y_predict =mdl.predict(X_predict)

	for r in range(0,m):
		if NP.isnan(X[r][-1]):
			X[r][-1] = y_predict[r]

	return X

if __name__ == '__main__':
	X = NP.array([[1, 2, 3], [2, 4, NP.nan], [3, 6, 9]])
	print 'Before:\n', X
	print 'After:\n', impute(X)




