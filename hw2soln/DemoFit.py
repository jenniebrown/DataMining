from VinoClassifier import load, train
from matplotlib import pyplot as plt

X_train,y_train = load('train.csv')
X_test,y_test = load('test.csv')

# number of neighbors (k of k-nn)
knns = [2**i for i in xrange(8)]

# average error with respect to training/test data
e_train = []
e_test = []

# build k-nn classifier for each k 
for k in knns:

    # your code here (hint: use "train" function in VinoClassifier and "score" function in KNeighborsClassifier) 
    clf = train(X_train, y_train, 'KNeighborsClassifier', n_neighbors = k)
    e_train.append(1-clf.score(X_train, y_train))
    e_test.append(1-clf.score(X_test, y_test))
        
# Plot average error of classifer with respect to training and test data
with plt.style.context('fivethirtyeight'):
    h_train, = plt.plot(knns, e_train, label = 'Training Error')
    h_test, = plt.plot(knns, e_test, label = 'Test Error') 

plt.xlabel('Number of Neighbors Used in Prediction')
plt.ylabel('Average Error')
plt.legend(handles = [h_train, h_test])
plt.show()

