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

    with open(file_name, 'r') as fp:

        # your code here

        X = []
        flag = True
        for line in fp:
            if flag:
                flag = False
                continue
            X.append([float(x) for x in line.strip().split(',')])
    
    X = np.array(X)
    X,y = X[:,:-1], X[:,-1]
    return X,y



def train(X, y, clf_name, **args):
    '''
    train data using given data

    input: 
        X: numpy array representing features
        y: numpy array representing target
        clf_name: name of classifier ('DecisionTreeClassifier', 'LinearSVC')
        args: list of classifier specific arguments, e.g., max_leaf_nodes
        
    output:
       classifier: trained classifier 
    '''

    CLF = globals()[clf_name]
    clf = CLF(**args)
    return clf.fit(X, y)


