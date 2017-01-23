from Classifier import load, loadTests, train, toFile, crossVal
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
from Features import getFeatures, cleanFeatures

# load data
X, y_train = load('train.csv')
Xt = load('test.csv')
yt = loadTests('test_label.csv')
X_trainInit = cleanFeatures(X)
X_testInit = cleanFeatures(Xt)

# number of neighbors (k of k-nn) and fitting parameters
a = 2 
c = 4
k = 123 

# Use cross-validation to estimate log loss of classifier
X_train = getFeatures(X_trainInit,a,c)
X_test = getFeatures(X_testInit,a,c)
scores = crossVal(X_train, y_train, 'KNeighborsClassifier',n_neighbors = k, weights = 'distance')
avg = np.mean(scores)
dev = scores.std()*2
print "Log loss = ",avg," stdev: ", dev, "Parameters a, c:",a, c

# Train classifier on full training set, predict class probabilities of the test set, and calculate accuracy.
clf = train(X_train, y_train, 'KNeighborsClassifier', n_neighbors = k,weights='distance')
P = clf.predict_proba(X_test)
acc = clf.score(X_test, yt)
test_scores = crossVal(X_test, yt, 'KNeighborsClassifier', n_neighbors = k, weights = 'distance')
print "Accuracy = ",acc

# Write probability results to file
result = []
i = 0
for [fail,success] in P:
	result.append([int(Xt[i,0]),fail,success])
	i += 1
toFile(result, 'knn_results.csv')


