# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:08:04 2015
# bug in mlpc can't refit with data new dims

Recommended:
TOXADJ=1
MLPC, kNN, RF, SVM, LSVC, LR, GA, ada_discrete

TOXADJ=0
ada_real, QDA
@author: 
"""
import math
import random
import numpy as np
import pandas as pd
#import Quandl
import pickle
import re
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25,\
                            maxCAR25
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, runsZScore, percentUpDays
from sklearn.preprocessing import scale, robust_scale, minmax_scale

import datetime
import time
import numpy as np
import pandas as pd
import sklearn

#from pandas_datareader import DataReader
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                        BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Perceptron, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
#from sklearn.neural_network import MLPClassifier

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from IPython.display import Image
import pydot

from sknn.mlp import Classifier, Layer
from sknn.backend import lasagne

path = '/media/sf_Python/data/from_fileprep/'
filename = 'F_ES_19981222to20160225signals.csv'
ticker = filename[:4]
#signal = 'ZZ 0.02,-0.005'
signal = 'volCheck'
#for matching columns in directional filter
dropCol = ['Open','High','Low','Close','Volume','gainAhead','signal','dates']
#            'Pri_RSI_Y1','Pri_RSI_Y2','Pri_RSI_Y3','Pri_RSI_Y4']

toxCheck = ['TOX20', 'TOX25']
#toxCheck = ['TOX25']
tox_adj_proportion = 1 # proportion of # -1's / #1's 

DD95_limit = 0.20
initial_equity = 100000.0
#dataLoadStartDate = "1998-12-22"
#dataLoadEndDate = "2016-01-04"

#offline learning
testFirstYear = "2008-01-01"
#adjust reserve for windowlength
testFinalYear = "2016-02-25"
test_split = 0.33 #test split
iterations = 10 #for sss

#online learning
validationFirstYear ="2015-01-01"
validationFinalYear ="2017-01-01"


RSILookback = 1.5
zScoreLookback = 10
ATRLookback = 5
DPOLookback = 3
# relATR -> 10 zscore atr lookback + shift 5 + 10zscore atr rel lookback
maxlb = max(RSILookback, zScoreLookback, ATRLookback, DPOLookback, 60)

start_time = time.time()
model_metrics = init_report()

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)

models = [#("GA_Reg", SymbolicRegressor(population_size=5000, generations=20,
          #                             tournament_size=20, stopping_criteria=0.0, 
          #                             const_range=(-1.0, 1.0), init_depth=(2, 6), 
          #                             init_method='half and half', transformer=True, 
          #                             comparison=True, trigonometric=True, 
          #                             metric='mean absolute error', parsimony_coefficient=0.001, 
          #                             p_crossover=0.9, p_subtree_mutation=0.01, 
          #                             p_hoist_mutation=0.01, p_point_mutation=0.01, 
          #                             p_point_replace=0.05, max_samples=1.0, 
          #                             n_jobs=1, verbose=0, random_state=None)),
         #("GA_Reg2", SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, comparison=True, transformer=False, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=1, verbose=0, parsimony_coefficient=0.01, random_state=0)),
         #("LR", LogisticRegression(class_weight={1:500})), \
         #("PRCEPT", Perceptron(class_weight={1:500})), \
         #("PAC", PassiveAggressiveClassifier(class_weight={1:500})), \
         #("LSVC", LinearSVC()), \
         #("GNBayes",GaussianNB()),\
         #("LDA", LinearDiscriminantAnalysis()), \
         #("QDA", QuadraticDiscriminantAnalysis()), \
         #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
         #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
         #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
         #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
         #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
         #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
         #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
         #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
         #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
         #("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
         #("RF", RandomForestClassifier(class_weight={1:500}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
         #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
         #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
         #uniform is faster
         #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
         #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
         ("VotingHard", VotingClassifier(estimators=[\
             ("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             ("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             ("QDA", QuadraticDiscriminantAnalysis()),\
             ("GNBayes",GaussianNB()),\
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
             #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             ("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                 ], voting='hard', weights=None)),
         #("VotingSoft", VotingClassifier(estimators=[\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("QDA", QuadraticDiscriminantAnalysis()),\
             #("GNBayes",GaussianNB()),\
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
             #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
         #        ], voting='soft', weights=None)),
         ]
         
qt = pd.read_csv(path+filename,parse_dates={'dates':['index']}, index_col='dates')
dataSet = pd.concat([qt[' OPEN'],qt[' HIGH'],qt[' LOW'],qt[' CLOSE'],qt[' VOL']], axis=1)
dataSet.columns = ['Open','High','Low','Close','Volume']
nrows = dataSet.shape[0]
print 'Successfully loaded ', nrows,' rows from ', filename

#dataSet.Close = qt.Close


dataSet['Pri_RSI'] = RSI(dataSet.Close,RSILookback)
dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)
dataSet['Pri_ATR'] = zScore(ATR(dataSet.High,dataSet.Low,dataSet.Close,ATRLookback),zScoreLookback)
dataSet['Pri_ATR_Y1'] = dataSet['Pri_ATR'].shift(1)
dataSet['Pri_ATR_Y2'] = dataSet['Pri_ATR'].shift(2)
dataSet['Pri_ATR_Y3'] = dataSet['Pri_ATR'].shift(3)
dataSet['Rel_ATR'] = zScore(ATR(dataSet.High,dataSet.Low,dataSet.Close,1)/dataSet['Pri_ATR'].shift(5),zScoreLookback)
dataSet['priceChange'] = priceChange(dataSet.Close)
dataSet['priceChangeY1'] = dataSet['priceChange'].shift(1)
dataSet['priceChangeY2'] = dataSet['priceChange'].shift(2)
dataSet['priceChangeY3'] = dataSet['priceChange'].shift(3)
dataSet['Pri_DPO'] = DPO(dataSet.Close,DPOLookback)
#dataSet['Pri_DPO_Y1'] = dataSet['Pri_DPO'].shift(1)
#dataSet['Pri_DPO_Y2'] = dataSet['Pri_DPO'].shift(2)
#dataSet['Pri_DPO_Y3'] = dataSet['Pri_DPO'].shift(3)
dataSet['GARCH'] = zScore(priceChange(garch(dataSet.priceChange)),zScoreLookback)
dataSet['GARCH_Y1'] = dataSet['GARCH'].shift(1)
dataSet['GARCH_Y2'] = dataSet['GARCH'].shift(2)
dataSet['GARCH_Y3'] = dataSet['GARCH'].shift(3)
dataSet['autoCor'] = autocorrel(dataSet.Close,3)
dataSet['autoCor_Y1'] = dataSet['autoCor'].shift(1)
dataSet['autoCor_Y2'] = dataSet['autoCor'].shift(2)
dataSet['autoCor_Y3'] = dataSet['autoCor'].shift(3)
dataSet['K_Eff'] = zScore(kaufman_efficiency(dataSet.Close,10),zScoreLookback)# 10 = 2 weeks
dataSet['K_Eff_Y1'] = dataSet['K_Eff'].shift(1)
dataSet['K_Eff_Y2'] = dataSet['K_Eff'].shift(2)
dataSet['K_Eff_Y3'] = dataSet['K_Eff'].shift(3)
dataSet['volSpike'] = zScore(volumeSpike(dataSet.Volume, 5), zScoreLookback)
dataSet['volSpike_Y1'] = dataSet['volSpike'].shift(1)
dataSet['volSpike_Y2'] = dataSet['volSpike'].shift(2)
dataSet['volSpike_Y3'] = dataSet['volSpike'].shift(3)
#dataSet['F_ED_zScore'] = zScore(priceChange(qt['F_ED']),30) # -inf if too many zeros
#dataSet['F_ED_zScore_Y1'] = dataSet['F_ED_zScore'].shift(1)
#dataSet['F_ED_zScore_Y2'] = dataSet['F_ED_zScore'].shift(2)
#dataSet['F_ED_zScore_Y3'] = dataSet['F_ED_zScore'].shift(3)
#dataSet['F_GC_zScore'] = zScore(priceChange(qt['F_GC']),zScoreLookback)
#dataSet['F_GC_zScore_Y1'] = dataSet['F_GC_zScore'].shift(1)
#dataSet['F_GC_zScore_Y2'] = dataSet['F_GC_zScore'].shift(2)
#dataSet['F_GC_zScore_Y3'] = dataSet['F_GC_zScore'].shift(3)
#dataSet['F_CL_zScore'] = zScore(priceChange(qt['F_CL']),zScoreLookback)
#dataSet['F_CL_zScore_Y1'] = dataSet['F_CL_zScore'].shift(1)
#dataSet['F_CL_zScore_Y2'] = dataSet['F_CL_zScore'].shift(2)
#dataSet['F_CL_zScore_Y3'] = dataSet['F_CL_zScore'].shift(3)
#dataSet['F_DX_zScore'] = zScore(priceChange(qt['F_DX']),zScoreLookback)
#dataSet['F_DX_zScore_Y1'] = dataSet['F_DX_zScore'].shift(1)
#dataSet['F_DX_zScore_Y2'] = dataSet['F_DX_zScore'].shift(2)
#dataSet['F_DX_zScore_Y3'] = dataSet['F_DX_zScore'].shift(3)
#dataSet['F_ED_WOWzScore'] = (qt['F_ED']-qt['F_ED'].shift(5))/qt['F_ED'].shift(5)
#dataSet['F_GC_WOWzScore'] = (qt['F_GC']-qt['F_GC'].shift(5))/qt['F_GC'].shift(5)
#dataSet['F_CL_WOWzScore'] = (qt['F_CL']-qt['F_CL'].shift(5))/qt['F_CL'].shift(5)
#dataSet['F_DX_WOWzScore'] = (qt['F_DX']-qt['F_DX'].shift(5))/qt['F_DX'].shift(5)

#add stocks
for col in qt.drop([ticker],axis=1).columns:
    if col[:2] =='S_' or col[:2] =='F_':
        dataSet[col+'_WOWzScore'] = (qt[col]-qt[col].shift(5))/qt[col].shift(5)
        #dataSet[col+'_zScore_Y1'] = dataSet[col+'_zScore'].shift(1)
        #dataSet[col+'_zScore_Y2'] = dataSet[col+'_zScore'].shift(2)
        #dataSet[col+'_zScore_Y3'] = dataSet[col+'_zScore'].shift(3)


dataSet['runsScore10'] = runsZScore(dataSet.priceChange,10)
dataSet['percentUpDays10'] = percentUpDays(dataSet.priceChange,10)
dataSet['mean60_ga'] = pd.rolling_mean(dataSet.priceChange,60)
dataSet['std60_ga'] = pd.rolling_std(dataSet.priceChange,60)
dataSet['kurt60_ga'] = pd.rolling_kurt(dataSet.priceChange,60)
dataSet['skew60_ga'] = pd.rolling_skew(dataSet.priceChange,60)
dataSet['gainAhead'] = gainAhead(dataSet.Close)
#qt['beLong'] = np.where(dataSet.gainAhead>beLongThreshold,1,-1)
#qt['beShort'] = np.where(dataSet.gainAhead<beShortThreshold,1,-1)
#
#if signal == 'beShort':
#    signal += '<'+str(beShortThreshold)
#    
#if signal == 'beLong':
#    signal += '>'+str(beLongThreshold)

#check dataSet for nan/inf beyond initial lookback periods maxlb
for col in dataSet:
    if sum(np.isnan(dataSet[col][maxlb:].values))>0:
        print dataSet[col][maxlb:][np.isnan(dataSet[col][maxlb:].values)]
        raise ValueError, 'nan in %s' % col
    elif sum(np.isinf(dataSet[col][maxlb:].values))>0:
        print dataSet[col][maxlb:][np.isnan(dataSet[col][maxlb:].values)]
        raise ValueError, 'inf in %s' % col
    elif sum(np.isneginf(dataSet[col][maxlb:].values))>0:
        print dataSet[col][maxlb:][np.isnan(dataSet[col][maxlb:].values)]
        raise ValueError, '-inf in %s' % col

#get tox trades    
checkVol = getToxCDF(abs(dataSet['gainAhead'].ix[testFirstYear:testFinalYear]))
for k in checkVol.keys():
    if k not in toxCheck:
        checkVol.pop(k)
#bestModelLong = {}
#bestModelShort = {}
for i in range(0,2):
    tox_adj_proportion = i
    for k,volThreshold in checkVol.iteritems():
        signal = k+'>'+str(volThreshold)
        qt[signal] = np.where(abs(dataSet.gainAhead)>volThreshold,1,-1)
        dataSet['signal'] = qt[signal]
           
        #  Select the date range to test
        mmData = dataSet.ix[testFirstYear:testFinalYear].dropna().reset_index()
        #datay_gainAhead = dataSet['gainAhead'].ix[testFirstYear:testFinalYear]
        datay_gainAhead_all = mmData[:-1].gainAhead
        nrows = mmData.shape[0]
        print "\nTesting Training Set %i rows.." % nrows
        mmData_adj = adjustDataProportion(mmData.iloc[:-1], tox_adj_proportion)  #drop last row for hold days =1
        nrows = mmData_adj.shape[0]
        datay = mmData_adj.signal
        datay_gainAhead = mmData_adj.gainAhead
        
        dataX = mmData_adj.drop(dropCol, axis=1) 
                                
        cols = dataX.columns.shape[0]
        print '\nTraining using %i features: ' % cols
        for i,x in enumerate(dataX.columns):
            print i,x+',',
            
        #  Copy from pandas dataframe to numpy arrays
        dy = np.zeros_like(datay)
        dX = np.zeros_like(dataX)
        
        dy = datay.values
        dX = dataX.values
        
        metaData = {'ticker':ticker, 'date__start':testFirstYear, 'date_end':testFinalYear,\
                 'signal':signal, 'rows':nrows,'cols':cols, \
                 'test_split':test_split, 'iters':iterations, 'tox_adj':tox_adj_proportion}
    
        #  Make 'iterations' index vectors for the train-test split
        sss = StratifiedShuffleSplit(dy,iterations,test_size=test_split, random_state=None)
        #fit & predict    
        for i,m in enumerate(models):
            cm_y_train = np.array([])
            cm_y_test = np.array([])
            cm_y_pred_is = np.array([])
            cm_y_pred_oos = np.array([])
            cm_train_index = np.array([],dtype=int)
            cm_test_index = np.array([],dtype=int)
            
            print '\n\nNew SSS train/predict loop for', m[0]
            for train_index,test_index in sss:
                X_train, X_test = dX[train_index], dX[test_index]
                y_train, y_test = dy[train_index], dy[test_index]
            
                #check if there are no intersections
                intersect = np.intersect1d(datay.reset_index().iloc[test_index]['index'].values,\
                            datay.reset_index().iloc[train_index]['index'].values)
                if intersect.size != 0:
                    print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
           
                #  fit the model to the in-sample data
                m[1].fit(X_train, y_train)
                #trained_models[m[0]] = pickle.dumps(m[1])
                            
                y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))
                #test on all the data that excludes those in X_train, hold period [:-1]
                X_test_all = mmData.iloc[:-1].drop(dataX.reset_index()\
                            .iloc[train_index]['index'].values, axis=0)\
                            .drop(dropCol,axis=1).values
                y_test_all = mmData.iloc[:-1].drop(datay.reset_index()\
                            .iloc[train_index]['index'].values, axis=0).signal.values
                test_index_all = mmData.iloc[:-1].drop(datay.reset_index().iloc[train_index]['index'].values, axis=0).index.values
                
                y_pred_oos = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_test_all)]))
    
                if m[0][:2] == 'GA':
                    print '\nProgram:', m[1]._program
                    print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
                
                cm_y_train = np.concatenate([cm_y_train,y_train])
                cm_y_test = np.concatenate([cm_y_test,y_test_all])
                cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
                cm_y_pred_oos = np.concatenate([cm_y_pred_oos,y_pred_oos])
                cm_train_index = np.concatenate([cm_train_index,train_index])
                cm_test_index = np.concatenate([cm_test_index,test_index_all])
            
            
            is_display_cmatrix2(cm_y_train, cm_y_pred_is, datay_gainAhead, cm_train_index, m[0], ticker, testFirstYear, testFinalYear, iterations, signal)
            #CAR25_L1_is = CAR25(signal, cm_y_pred_is, datay.reset_index().iloc[train_index]['index'].values, mmData.Close, 'LONG', 1)
            #CAR25_Ln1_is = CAR25(signal, cm_y_pred_is, datay.reset_index().iloc[train_index]['index'].values, mmData.Close, 'LONG', -1)
            #CAR25_S1_is = CAR25(signal, cm_y_pred_is, datay.reset_index().iloc[train_index]['index'].values, mmData.Close, 'SHORT', 1)
            #CAR25_Sn1_is = CAR25(signal, cm_y_pred_is, datay.reset_index().iloc[train_index]['index'].values, mmData.Close, 'SHORT', -1)
            #CAR25_is_list = [CAR25_L1_is, CAR25_Ln1_is, CAR25_S1_is, CAR25_Sn1_is]
            #for c in CAR25_is_list:
            model_metrics = update_report(model_metrics, "IS", cm_y_pred_is, cm_y_train, datay_gainAhead, cm_train_index, m, metaData)
            
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead_all, cm_test_index, m[0], ticker, testFirstYear, testFinalYear, iterations, signal)
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, datay.reset_index().iloc[test_index]['index'].values, mmData.Close, 'LONG', 1)
            #CAR25_Ln1_oos = CAR25(signal, cm_y_pred_oos, datay.reset_index().iloc[test_index]['index'].values, mmData.Close, 'LONG', -1)
            #CAR25_S1_oos = CAR25(signal, cm_y_pred_oos, datay.reset_index().iloc[test_index]['index'].values, mmData.Close, 'SHORT', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, datay.reset_index().iloc[test_index]['index'].values, mmData.Close, 'SHORT', -1)
            #CAR25_oos_list = [CAR25_L1_oos, CAR25_Ln1_oos, CAR25_S1_oos, CAR25_Sn1_oos]
            #for c in CAR25_oos_list:
            model_metrics = update_report(model_metrics, "OOS", cm_y_pred_oos, cm_y_test, datay_gainAhead_all, cm_test_index, m, metaData)
                
            #CAR25_MAX_oos = maxCAR25(CAR25_oos_list) 
            #sss_display_cmatrix(cm_is, cm_oos, m[0], ticker, testFirstYear, testFinalYear, iterations, signal)
                    
            #if CAR25_MAX_oos['C25sig'][:4] =='LONG':
            #    if m[0] not in bestModelLong:
            #        bestModelLong[m[0]] = [CAR25_MAX_oos, metaData,m]
            #    else:
            #        if bestModelLong[m[0]][0]['CAR25'] <CAR25_MAX_oos['CAR25']:
            #            bestModelLong[m[0]] = [CAR25_MAX_oos, metaData,m]
            #else:
            #    if m[0] not in bestModelShort:
            #        bestModelShort[m[0]] = [CAR25_MAX_oos, metaData,m]
            #    else:
            #        if bestModelShort[m[0]][0]['CAR25'] <CAR25_MAX_oos['CAR25']:
            #            bestModelShort[m[0]] = [CAR25_MAX_oos, metaData,m]
                    
#filename_BML = re.sub(r'[^\w]', '', str(datetime.datetime.today())) + '_VF_BModelLong.pkl'
#filename_BMS = re.sub(r'[^\w]', '', str(datetime.datetime.today())) + '_VF_BModelShort.pkl'

#find and save best volatility model
model_score_oos = model_metrics.loc[model_metrics['sample'] == 'OOS'].reset_index()
#zscore
model_score_oos['f1mm'] =minmax_scale(model_score_oos.f1.reshape(-1, 1))
model_score_oos['accmm'] =minmax_scale(model_score_oos.acc.reshape(-1, 1))
model_score_oos['precmm'] =minmax_scale(model_score_oos.prec.reshape(-1, 1))
model_score_oos['recmm'] = minmax_scale(model_score_oos.rec.reshape(-1, 1))
model_score_oos['fn_magmm'] =-minmax_scale(model_score_oos.fn_mag.reshape(-1, 1))
model_score_oos['fp_magmm'] =-minmax_scale(model_score_oos.fp_mag.reshape(-1, 1))
model_score_oos['scoremm'] =  model_score_oos.f1mm+model_score_oos.accmm+model_score_oos.precmm+\
                                model_score_oos.recmm+model_score_oos.fn_magmm+model_score_oos.fp_magmm

#model_score_oos['CAR25zs'] =(model_score_oos['CAR25']-model_score_oos['CAR25'].mean())/model_score_oos.CAR25.std()
#model_score_oos['DD100zs'] =-(model_score_oos['DD100']-model_score_oos['DD100'].mean())/model_score_oos.DD100.std()
#model_score_oos['SOR25zs'] =(model_score_oos['SOR25']-model_score_oos['SOR25'].mean())/model_score_oos.SOR25.std()
#model_score_oos['TPYzs'] =(model_score_oos['TPY']-model_score_oos['TPY'].mean())/model_score_oos.TPY.std()
#model_score_oos['rowszs'] = (model_score_oos['rows']-model_score_oos['rows'].mean())/model_score_oos.rows.std()
#model_score_oos['scorezs'] =  model_score_oos.f1zs+model_score_oos.CAR25zs+model_score_oos.DD100zs+model_score_oos.SOR25zs+model_score_oos.TPYzs

#softmax
model_score_oos['f1sm'] = softmax_score(model_score_oos['f1'])
model_score_oos['accsm'] = softmax_score(model_score_oos['acc'])
model_score_oos['precsm'] = softmax_score(model_score_oos['prec'])
model_score_oos['recsm'] = softmax_score(model_score_oos['rec'])
model_score_oos['fn_magsm'] =-softmax_score(minmax_scale(model_score_oos['fn_mag']))
model_score_oos['fp_magsm'] =-softmax_score(minmax_scale(model_score_oos['fp_mag']))
model_score_oos['scoresm'] =  model_score_oos.f1sm+model_score_oos.accsm\
            +model_score_oos.precsm+model_score_oos.recsm+model_score_oos.fn_magsm+model_score_oos.fp_magsm

#model_score_oos['CAR25sm'] =softmax_score(model_score_oos['CAR25'])
#model_score_oos['DD100sm'] =-softmax_score(model_score_oos['DD100'])
#model_score_oos['SOR25sm'] =softmax_score(model_score_oos['SOR25'])
#model_score_oos['TPYsm'] =softmax_score(model_score_oos['TPY'])
#model_score_oos['rowssm'] = softmax_score(model_score_oos['rows'])
#model_score_oos['scoresm'] =  model_score_oos.f1sm+model_score_oos.CAR25sm+model_score_oos.DD100sm+model_score_oos.SOR25sm+model_score_oos.TPYsm+model_score_oos.rowssm

#zscore                        
model_score_is = model_metrics.loc[model_metrics['sample'] == 'IS'].reset_index()
model_score_is['f1mm'] =minmax_scale(model_score_is.f1.reshape(-1, 1))
model_score_is['accmm'] =minmax_scale(model_score_is.acc.reshape(-1, 1))
model_score_is['precmm'] =minmax_scale(model_score_is.prec.reshape(-1, 1))
model_score_is['recmm'] = minmax_scale(model_score_is.rec.reshape(-1, 1))
model_score_is['fn_magmm'] =-minmax_scale(model_score_is.fn_mag.reshape(-1, 1))
model_score_is['fp_magmm'] = -minmax_scale(model_score_is.fp_mag.reshape(-1, 1))
model_score_is['scoremm'] =  model_score_is.f1mm+model_score_is.accmm+model_score_is.precmm+\
                                model_score_is.recmm+model_score_is.fn_magmm+model_score_is.fp_magmm

#model_score_is['CAR25zs'] =(model_score_is['CAR25']-model_score_is['CAR25'].mean())/model_score_is.CAR25.std()
#model_score_is['DD100zs'] =-(model_score_is['DD100']-model_score_is['DD100'].mean())/model_score_is.DD100.std()
#model_score_is['SOR25zs'] =(model_score_is['SOR25']-model_score_is['SOR25'].mean())/model_score_is.SOR25.std()
#model_score_is['TPYzs'] =(model_score_is['TPY']-model_score_is['TPY'].mean())/model_score_is.TPY.std()
#model_score_is['rowszs'] = (model_score_is['rows']-model_score_is['rows'].mean())/model_score_is.rows.std()
#model_score_is['scorezs'] =  model_score_is.f1zs+model_score_is.CAR25zs+model_score_is.DD100zs+model_score_is.SOR25zs+model_score_is.TPYzs

#sm
model_score_is['f1sm'] = softmax_score(model_score_is['f1'])
model_score_is['accsm'] = softmax_score(model_score_is['acc'])
model_score_is['precsm'] = softmax_score(model_score_is['prec'])
model_score_is['recsm'] = softmax_score(model_score_is['rec'])
model_score_is['fn_magsm'] =-softmax_score(minmax_scale(model_score_is['fn_mag']))
model_score_is['fp_magsm'] =-softmax_score(minmax_scale(model_score_is['fp_mag']))
model_score_is['scoresm'] =  model_score_is.f1sm+model_score_is.accsm\
            +model_score_is.precsm+model_score_is.recsm+model_score_is.fn_magsm+model_score_is.fp_magsm
            
#model_score_is['CAR25sm'] =softmax_score(model_score_is['CAR25'])
#model_score_is['DD100sm'] =-softmax_score(model_score_is['DD100'])
#model_score_is['SOR25sm'] =softmax_score(model_score_is['SOR25'])
#model_score_is['TPYsm'] =softmax_score(model_score_is['TPY'])
#model_score_is['rowssm'] = softmax_score(model_score_is['rows'])
#model_score_is['scoresm'] =  model_score_is.f1sm+model_score_is.CAR25sm+model_score_is.DD100sm+model_score_is.SOR25sm+model_score_is.TPYsm+model_score_is.rowssm

#minimum of both scores to find best model
combined_scoremm = pd.Series(minmax_scale(pd.concat([model_score_is['scoremm'],model_score_oos['scoremm']], axis=1).min(axis=1)))
combined_scoremm.name = 'c_scoremm'
combined_scoresm = pd.Series(minmax_scale(pd.concat([model_score_is['scoresm'],model_score_oos['scoresm']], axis=1).min(axis=1)))
combined_scoresm.name = 'c_scoresm'
combined_final = combined_scoremm * combined_scoresm
combined_final.name = 'final_score'
c_score_df = pd.concat([pd.concat([combined_final,combined_final],axis=0),\
            pd.concat([combined_scoremm,combined_scoremm],axis=0),\
            pd.concat([combined_scoresm,combined_scoresm],axis=0)], axis=1)
#c_score_df.columns = ['c_scorezs','c_scoresm']


#save metrics
filename_metrics = re.sub(r'[^\w]', '', str(datetime.datetime.today())) + '_find_vol.csv'
pd.concat([c_score_df, pd.concat([model_score_is,model_score_oos], axis=0)],axis=1)\
            .sort_values(['final_score'], ascending=False)\
            .to_csv('/media/sf_Python/'+filename_metrics)

#retrain best model
bestModel = model_score_oos.iloc[combined_final.idxmax()]
tox_adj_proportion = bestModel.tox_adj
for m in models:
    if bestModel['params'] == str(m[1]):
        print  'Best model found...\n', m[1]
        bm = m[1]
        
volThreshold = checkVol[bestModel['signal'][:5]]
dataSet['signal'] = np.where(abs(dataSet.gainAhead)>volThreshold,1,-1)

#  Select the date range to test
mmData = dataSet.ix[testFirstYear:testFinalYear].dropna().reset_index()

print "\nTraining %i rows for Best Model.." % nrows
mmData_adj = adjustDataProportion(mmData.iloc[:-1], tox_adj_proportion)  #drop last row for hold days =1
#nrows = mmData_adj.shape[0]
datay = mmData_adj.signal
datay_gainAhead = mmData_adj.gainAhead

dataX = mmData_adj.drop(dropCol, axis=1)                       
    
#  Copy from pandas dataframe to numpy arrays
dy = np.zeros_like(datay)
dX = np.zeros_like(dataX)

dy = datay.values
dX = dataX.values
        
bm.fit(dX, dy)
y_pred_is = np.array(([-1 if x<0 else 1 for x in bm.predict(dX)]))
is_display_cmatrix2(dy, y_pred_is, datay_gainAhead, dataX.reset_index().index,\
     bm, ticker, testFirstYear, testFinalYear, 1, bestModel['signal'], 1)
     
#if  bestModel['C25sig'][0] == 'L':
#    direction = 'LONG'
#else:
#    direction = 'SHORT'
#CAR25(bestModel['signal'], y_pred_is, datay.reset_index().iloc[dataX.reset_index().index]['index'].values,\
#        mmData.Close, direction, int(bestModel['C25sig'][-2:]))
CAR25(bestModel['signal'], y_pred_is, datay.reset_index().iloc[dataX.reset_index().index]['index'].values, mmData.Close, 'LONG', 1)
CAR25(bestModel['signal'], y_pred_is, datay.reset_index().iloc[dataX.reset_index().index]['index'].values, mmData.Close, 'LONG', -1)
CAR25(bestModel['signal'], y_pred_is, datay.reset_index().iloc[dataX.reset_index().index]['index'].values, mmData.Close, 'SHORT', 1)
CAR25(bestModel['signal'], y_pred_is, datay.reset_index().iloc[dataX.reset_index().index]['index'].values, mmData.Close, 'SHORT', -1)

mmData_v = dataSet.ix[validationFirstYear:validationFinalYear].dropna().reset_index()

datay_v = mmData_v.signal
datay_gainAhead_v = mmData_v.gainAhead

dataX_v = mmData_v.drop(dropCol, axis=1) 
cols_v = dataX_v.columns.shape[0]
nrows_v = mmData_v.shape[0]

print "Evaluating on Validation Set %i rows.." % nrows_v

#  Reset dX,Dy
dy_v = np.zeros_like(datay_v)
dX_v = np.zeros_like(dataX_v)

dy_v = datay_v.values
dX_v = dataX_v.values

y_pred_oos_v = np.array(([-1 if x<0 else 1 for x in bm.predict(dX_v)]))
oos_display_cmatrix2(dy_v,y_pred_oos_v,datay_gainAhead_v, mmData_v.index, bm, ticker,validationFirstYear, validationFinalYear, 1, bestModel['signal'],1)
#CAR25(bestModel['signal'], y_pred_oos_v, datay_v.index.values, mmData_v.Close, direction, int(bestModel['C25sig'][-2:]))
CAR25(bestModel['signal'], y_pred_oos_v, datay_v.index.values, mmData_v.Close, 'LONG', 1)
CAR25(bestModel['signal'], y_pred_oos_v, datay_v.index.values, mmData_v.Close, 'LONG', -1)
CAR25(bestModel['signal'], y_pred_oos_v, datay_v.index.values, mmData_v.Close, 'SHORT', 1)
CAR25(bestModel['signal'], y_pred_oos_v, datay_v.index.values, mmData_v.Close, 'SHORT', -1)

#bestModel = bestModel.drop(['index','sample',u'f1', u'acc', u'prec', u'rec', u'safef',
#        'CAR50', 'CAR75', 'DD95', 'DD100', 'SOR25', 'SHA25',
#        'avg_prec', 'logloss', 'ham', 'tp', 'fn', 'fp',
#        'tn', 'test_split', 'iters', 'params',
#        'CAR25', 'YIF', 'TPY', 'rows', 'cols', 'f1zs', 'rowszs',
#        'CAR25zs', 'DD100zs', 'SOR25zs', 'TPYzs', 'scorezs',
#        'f1sm', 'rowssm','CAR25sm', 'DD100sm', 'SOR25sm', 'TPYsm', 'scoresm'])

for col in bestModel.axes[0]:   
    if col == 'ticker' or col == 'model' or col == 'signal' or col == 'date__start' or col == 'date_end' or col == 'C25sig' or col == 'tox_adj':
        #print col        
        pass
    else:
        bestModel = bestModel.drop([col])
        
print 'Saving Model..'
filename_bm = 'VF_'+ re.sub(r'[^\w]', '', str(datetime.datetime.today())) + bestModel['model'] +\
                    re.sub(r'[^\w]', '', str(bestModel)) +'.pkl'
                    
if bestModel['model'][:2] == 'GA':
    print '\nProgram:', bm._program
    print 'R^2:    ', bm.score(X_test_all,y_test_all) 
    joblib.dump(bm._program, '/media/sf_Python/saved_models/VF/'+filename_bm, compress=True)
else:
    joblib.dump(bm, '/media/sf_Python/saved_models/VF/'+filename_bm, compress=True)

#for mod in bestModelLong:
#    filename_BML = re.sub(r'[^\w]', '', str(datetime.datetime.today())) + mod +\
#            re.sub(r'[^\w]', '', str(bestModelLong[mod][1])) +'_VF_BModelLong.pkl'
#    joblib.dump(bestModelLong[mod][2][1], '/media/sf_Python/'+filename_BML)
#for mod in bestModelShort:
#    filename_BMS = re.sub(r'[^\w]', '', str(datetime.datetime.today())) + mod +\
#            re.sub(r'[^\w]', '', str(bestModelShort[mod][1])) +'_VF_BModelShort.pkl'
#    joblib.dump(bestModelShort[mod][2][1], '/media/sf_Python/'+filename_BMS)

#save datafile for directional model training
dataX = dataSet.reset_index().drop(dropCol, axis=1)
dX = np.zeros_like(dataX.dropna().values)
dX = dataX.dropna().values
y_pred_is = np.array(([-1 if x<0 else 1 for x in bm.predict(dX)]))

dataX = dataSet.drop(['Open','High','Low','Close',
                       'Volume','signal'],
                        axis=1)

mmData = pd.concat([qt,dataX], axis=1).dropna().reset_index().drop(['Unnamed: 0','level_0'], axis=1)

filename_df = ticker+'_'+ bestModel['signal'][:5] +'_'+  bestModel['model'] +'_v'+ validationFirstYear + \
            'to' +validationFinalYear+'_'+re.sub(r'[^\w]', '', str(datetime.datetime.today()))[:14] +'.csv'

filename_dps = ticker+'_'+ bestModel['signal'][:5] +'_'+ bestModel['model']+'_v'+ validationFirstYear + \
            'to' +validationFinalYear+'_'+re.sub(r'[^\w]', '', str(datetime.datetime.today()))[:14] +'.csv'

mmData.to_csv('/media/sf_Python/data/from_vf_to_df/OHLCV_'+filename_df, index=False)
mmData.drop(mmData.index[np.where(y_pred_is == 1)]).to_csv('/media/sf_Python/data/from_vf_to_df/LV_'+filename_df)
mmData.drop(mmData.index[np.where(y_pred_is == -1)]).to_csv('/media/sf_Python/data/from_vf_to_df/HV_'+filename_df)
pd.concat([pd.Series(data=y_pred_is, index=mmData.index, name='signals'),\
     mmData.gainAhead], axis=1).set_index(pd.DatetimeIndex(mmData.dates)).ix[testFinalYear:validationFinalYear]\
     .to_csv('/media/sf_Python/data/from_vf_to_dps/SST_'+filename_dps)
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
print 'Next Step: 1) Run DPS on VF filtered  2) train DF filter'
