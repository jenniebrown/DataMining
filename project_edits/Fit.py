from Classifier import load, loadTests, train, toFile, crossVal
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
from Features import getFeatures, cleanFeatures
X, y_train = load('train.csv')
Xt = load('test.csv')
yt = loadTests('test_label.csv')
X_trainInit = cleanFeatures(X)
X_testInit = cleanFeatures(Xt)
# number of neighbors (k of k-nn)
knns = [2**i for i in xrange(2,11)]
#C = [i for i in xrange(1,5)]
C = [4]
#A = [i for i in xrange(1,4)]
A = [2]
a = 2 
c =4 

#knns = [i for i in xrange(120,130)]
k = 123 
# average error with respect to training/test data
e_train = []
e_test = []

# build k-nn classifier for each k
#for c in C:
#	for a in A:
#		X_train, X_test = getFeatures(X_trainInit, X_testInit,a,c)
#		scores = crossVal(X_train, y_trainInit, 'KNeighborsClassifier', n_neighbors = k, weights = 'distance')
#		avg = np.mean(scores)
#		print avg, a, c
X_train = getFeatures(X_trainInit,a,c)
X_test = getFeatures(X_testInit,a,c)
scores = crossVal(X_train, y_train, 'KNeighborsClassifier',n_neighbors = k, weights = 'distance')
avg = np.mean(scores)
dev = scores.std()*2
print "Log loss = ",avg," std: ", dev, "Parameters a, c:",a, c


clf = train(X_train, y_train, 'KNeighborsClassifier', n_neighbors = k,weights='distance')
P = clf.predict_proba(X_test)
acc = clf.score(X_test, yt)
test_scores = crossVal(X_test, yt, 'KNeighborsClassifier', n_neighbors = k, weights = 'distance')
print "Test log loss = ",np.mean(test_scores)
print "Accuracy = ",acc
i = 0
result = []
for [fail,success] in P:
	result.append([int(Xt[i,0]),fail,success])
	i += 1
toFile(result, 'knn_results.csv')
e_train.append(1-clf.score(X_train, y_train))

# build SVM classifier

# Plot average error of classifier with respect to training and test data
#with plt.style.context('fivethirtyeight'):
#	h_train, = plt.plot(knns, e_train, label = 'Training Error')
#
#plt.xlabel('Number of Neighbors Used in Prediction')
#plt.ylabel('Average Error')
#plt.show()


