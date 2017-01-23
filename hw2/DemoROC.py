'''
Jennifer Brown
jlb315
CSE 347
HW 2
DemoROC
'''
from VinoClassifier import load, train
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

X_train,y_train = load('train.csv')
X_test,y_test = load('test.csv')

# compute prediction probability
clf = train(X_train, y_train, 'KNeighborsClassifier', n_neighbors = 64)
P = clf.predict_proba(X_test)
#print P
# replace first column of P with y_test
P[:,0] = y_test
#print P
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
    #print pred_pos_inst, true_pos_inst
    pred_neg_inst = total_inst-pred_pos_inst
    true_neg_inst = pred_neg_inst - true_pos_inst
    false_neg_inst = pred_neg_inst - true_neg_inst
    false_pos_inst = pred_pos_inst - true_pos_inst
    tprs.append(true_pos_inst/(true_pos_inst+false_neg_inst))
    fprs.append((false_pos_inst)/(true_neg_inst+false_pos_inst))
    # your code here!!! 
    # hints: 
    #     fill up tprs and fprs
    
  

# Plot average error of classifer with respect to training and test data
#fig = plt.figure()
with plt.style.context('fivethirtyeight'):
    h_random, = plt.plot(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05), label = 'Random Guess')
    h_clf, = plt.plot(fprs, tprs, label = 'KNN Classifier') 

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(handles = [h_random, h_clf])
plt.show()
#fig.savefig('ROC.png')
