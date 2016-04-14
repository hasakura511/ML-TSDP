import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import cluster, covariance, manifold
from sklearn.qda import QDA

import operator
import re
from dateutil import parser

import datetime

import numpy as np
import matplotlib.pyplot as plt
try:
    from matplotlib.finance import quotes_historical_yahoo
except ImportError:
    from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

from os import listdir
from os.path import isfile, join
import re
import pandas as pd

savePath='./p/params/'
def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe 
    label the categories and split into train and test
    """
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_Out']
    dataset.ix[dataset.UpDown >= 0,'UpDown'] = 'Up'
    dataset.ix[dataset.UpDown < 0,'UpDown'] = 'Down'
    #dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    #dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]              
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test   
    
def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
    """
    performs classification on daily returns using several algorithms (method).
    method --> string algorithm
    parameters --> list of parameters passed to the classifier (if any)
    fout --> string with name of stock to be predicted
    savemodel --> boolean. If TRUE saves the model to pickle file
    """
   
    if method == 'RF':   
        return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'SVM':   
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'GTB': 
        return performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'QDA': 
        return performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
def performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    global savePath
    print 'RFClass ' 
    
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    if savemodel == True:
        fname_out = savePath + '{}-{}.pickle'.format(fout, datetime.datetime.now())
        if len(parameters) > 0:
            fname_out=parameters[0]
        
        print 'RFClass Saving ' + fname_out
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    accuracy = clf.score(X_test, y_test)
    print 'Accuracy:',accuracy
    return clf
    
def performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.datetime.now())
        with open(savePath + fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    print 'Accuracy:',accuracy
    return clf
    
def performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    SVM binary Classification
    """
    c = parameters[0]
    g =  parameters[1]
    clf = SVC()
    clf.fit(X_train, y_train)
 
    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.datetime.now())
        with open(savePath + fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    print 'Accuracy:',accuracy
    return clf
    
def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Ada Boosting binary Classification
    """
    n = parameters[0]
    l =  parameters[1]
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.datetime.now())
        with open(savePath + fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    print 'Accuracy:',accuracy
    return clf
    
def performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
 
    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.datetime.now())
        with open(savePath + fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    print 'Accuracy:',accuracy
    return clf
    
def performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Quadratic Discriminant Analysis binary Classification
    """
    def replaceTiny(x):
        if (abs(x) < 0.0001):
            x = 0.0001
    
    X_train = X_train.apply(replaceTiny)
    X_test = X_test.apply(replaceTiny)
    
    clf = QDA()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.datetime.now())
        with open(savePath + fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    print 'Accuracy:',accuracy
    return clf
