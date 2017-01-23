'''
Jennifer Brown
jlb315
CSE 347
HW2
DemoFit
'''

from VinoClassifier import load, train
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn

X_train,y_train = load('train.csv')
X_test,y_test = load('test.csv')

# number of neighbors (k of k-nn)
knns = [2**i for i in xrange(8)]

# average error with respect to training/test data
e_train = []
e_test = []


# build k-nn classifier for each k 
for k in knns:

    #     (1) fill up e_train and e_test 
    #     (2) use "train" function in VinoClassifier and "score" function in KNeighborsClassifier) 
    #
    mdl = train(X_train,y_train,'KNeighborsClassifier',n_neighbors = k)
    e_train.append(float(1-mdl.score(X_train,y_train)))
    mdl = train(X_test, y_test, 'KNeighborsClassifier',n_neighbors = k)
    e_test.append(float(1 - mdl.score(X_test,y_test)))
     
#Plot average error of classifer with respect to training and test data
#fig = plt.figure()
with plt.style.context('fivethirtyeight'):
    h_train, = plt.plot(knns, e_train, label = 'Training Error')
    h_test, = plt.plot(knns, e_test, label = 'Test Error') 

plt.xlabel('Number of Neighbors Used in Prediction')
plt.ylabel('Average Error')
plt.legend(handles = [h_train, h_test],loc=4)
#plt.show()
#fig.savefig('temp.png')

