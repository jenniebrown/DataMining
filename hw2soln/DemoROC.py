from VinoClassifier import load, train
from matplotlib import pyplot as plt
import numpy as np

X_train,y_train = load('train.csv')
X_test,y_test = load('test.csv')

# compute prediction probability
clf = train(X_train, y_train, 'KNeighborsClassifier', n_neighbors = 64)
P = clf.predict_proba(X_test)
print P
# replace first column of P with y_test
P[:,0] = y_test

# total number of instances
total_inst = P.shape[0]
# number of actual positive instances
total_pos_inst = np.count_nonzero(P[:,0])

# true positive rates and false positive rates
tprs,fprs = [],[]

# iterate over all positive thresholds
for threshold in np.unique(P[:,1]):
    
    # instances predicted to be positive
    pred_pos = P[P[:,1] >= threshold, :]
    # number of predicted positive instances
    pred_pos_inst = pred_pos.shape[0]
    # number of true positive instances
    true_pos_inst = np.count_nonzero(pred_pos[:,0])
      
    # your code here (hint: fill up tprs and fprs)
    tpr, fpr = 0, 0
    if total_pos_inst > 0:
        tpr = true_pos_inst*1.0/total_pos_inst
    tprs.append(tpr)

    if total_inst - total_pos_inst > 0:
        fpr = (pred_pos_inst - true_pos_inst)*1.0/(total_inst - total_pos_inst)    
    fprs.append(fpr)
  

# Plot average error of classifer with respect to training and test data
#with plt.style.context('fivethirtyeight'):
#    h_random, = plt.plot(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05), label = 'Random Guess')
#    h_clf, = plt.plot(fprs, tprs, label = 'KNN Classifier') 
#
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend(handles = [h_random, h_clf])
#plt.show()

