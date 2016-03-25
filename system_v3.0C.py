
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015
changelog
v3.1
model drop for v1

v3.0 "KOBE"
added validation period optimization
added more pairs
fixed offline mode
added minute version of CAR25
added timestamp, cycletime

@author: hidemi
"""
import math
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
from pytz import timezone
from datetime import datetime as dt
from os import listdir
from os.path import isfile, join

#classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                        BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

#regression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor,\
                        ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge                       
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import ARDRegression, Ridge,RANSACRegressor,\
                            LinearRegression, Lasso, LassoLars, BayesianRidge, PassiveAggressiveRegressor,\
                            SGDRegressor,TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import LinearSVR, SVR

#other
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
from sklearn.externals import joblib
import arch

#suztoolz
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve,\
                         directional_scoring, compareEquity, describeDistribution,\
                         offlineMode
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df_min,\
                            maxCAR25, wf_classify_validate, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS,\
                            calcEquity_df
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
from suztoolz.transform import zigzag as zg
from suztoolz.data import getDataFromIB

start_time = time.time()

#system parameters
version = 'v3'
version_ = 'v3.1'

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='1 min'
currencyPairs = ['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']

                            

def saveModel(dataSet):    
    mData = dataSet.drop(['Open','High','Low','Close',
                           'Volume','prior_index','gainAhead'],
                            axis=1).dropna()    
    #  Select the date range to test no label for the last index
    mmData = mData.iloc[-wf_is_period:-1]

    datay = mmData.signal
    mmData = mmData.drop(['signal'],axis=1)

    print '\nIn-Sample Period: %i ' % wf_is_period
    print 'Total %i features: ' % mmData.shape[1]
    for i,x in enumerate(mmData.columns):
        print i,x+',',
        #feature_names = feature_names+[x]    
    dataX = mmData

    #  Copy from pandas dataframe to numpy arrays
    dy = np.zeros_like(datay)
    dX = np.zeros_like(dataX)

    dy = datay.values
    dX = dataX.values

    #  Make 'iterations' index vectors for the train-test split
    sss = StratifiedShuffleSplit(dy,iterations,test_size=0.33,
                                 random_state=None)

    #  Initialize the confusion matrix
    cm_sum_is = np.zeros((2,2))
    cm_sum_oos = np.zeros((2,2))
        
    #  For each entry in the set of splits, fit and predict
    for train_index,test_index in sss:
        X_train, X_test = dX[train_index], dX[test_index]
        y_train, y_test = dy[train_index], dy[test_index] 

    #  fit the model to the in-sample data
        model.fit(X_train, y_train)

    #  test the in-sample fit    
        y_pred_is = model.predict(X_train)
        cm_is = confusion_matrix(y_train, y_pred_is)
        cm_sum_is = cm_sum_is + cm_is

    #  test the out-of-sample data
        y_pred_oos = model.predict(X_test)
        cm_oos = confusion_matrix(y_test, y_pred_oos)
        cm_sum_oos = cm_sum_oos + cm_oos

    tpIS = cm_sum_is[1,1]
    fnIS = cm_sum_is[1,0]
    fpIS = cm_sum_is[0,1]
    tnIS = cm_sum_is[0,0]
    precisionIS = tpIS/(tpIS+fpIS)
    recallIS = tpIS/(tpIS+fnIS)
    accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
    f1IS = (2.0 * precisionIS * recallIS) / (precisionIS+recallIS) 

    tpOOS = cm_sum_oos[1,1]
    fnOOS = cm_sum_oos[1,0]
    fpOOS = cm_sum_oos[0,1]
    tnOOS = cm_sum_oos[0,0]
    precisionOOS = tpOOS/(tpOOS+fpOOS)
    recallOOS = tpOOS/(tpOOS+fnOOS)
    accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
    f1OOS = (2.0 * precisionOOS * recallOOS) / (precisionOOS+recallOOS) 

    print "\n\nSymbol is ", ticker
    print "Learning algorithm is", model
    print "Confusion matrix for %i randomized tests" % iterations
    print "for years ", dataSet.index[0] , " through ", dataSet.index[-2]  

    print "\nIn sample"
    print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpIS, fnIS, recallIS)
    print "neg:  %i  %i" % (fpIS, tnIS)
    print "      %.2f          %.2f " % (precisionIS, accuracyIS)
    print "f1:   %.2f" % f1IS

    print "\nOut of sample"
    print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpOOS, fnOOS, recallOOS)
    print "neg:  %i  %i" % (fpOOS, tnOOS)
    print "      %.2f          %.2f " % (precisionOOS, accuracyOOS)
    print "f1:   %.2f" % f1OOS

    print "\nend of run"

    model.fit(dX, dy)
    
        
#no args -> debug.  else live mode arg 1 = pair, arg 2 = "0" to turn off
if len(sys.argv)==1:
    debug=True
    
    livePairs =  [
                    #'NZDJPY',\
                    #'CADJPY',\
                    #'CHFJPY',\
                    #'EURJPY',\
                    'GBPJPY',\
                    #'AUDJPY',\
                    #'USDJPY',\
                    #'AUDUSD',\
                    #'EURUSD',\
                    #'GBPUSD',\
                    #'USDCAD',\
                    #'USDCHF',\
                    #'NZDUSD',
                    #'EURCHF',\
                    #'EURGBP'\
                    ]
                    
                    
    showDist =  False
    showPDFCDF = False
    showAllCharts = True
    perturbData = True
    runDPS = True
    #scorePath = './debug/scored_metrics_'
    #equityStatsSavePath = './debug/'
    #signalPath = './debug/'
    #dataPath = './data/from_IB/'
    scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    modelPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/pickle/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    
else:
    print 'Live Mode', sys.argv[1], sys.argv[2]
    debug=False
    
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    runDPS = True
    
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    bestParamsPath =  './data/params/'
    
    if sys.argv[2] == "0":
        livePairs=[]
        ticker = sys.argv[1]
        offlineMode(ticker, "Offline Mode: "+sys.argv[0]+' '+sys.argv[1], signalPath, version, version_)
    else:
        livePairs=[sys.argv[1]]
           

        
for ticker in livePairs:
    print 'Begin optimization run for', ticker
    symbol=ticker[0:3]
    currency=ticker[3:6]

    #Model Parameters
    #dataSet length needs to be divisiable by each validation period! 
    validationSetLength = 1000
    #validationSetLength = 500
    #validationPeriods = [50,250]
    validationPeriods = [100,250] # min is 2
    #validationStartPoint = None
    #signal_types = ['gainAhead','ZZ']
    #signal_types = ['ZZ']
    signal_types = ['gainAhead']
    ga_steps = [30]
    zz_steps = [0.001,0.002,0.003]
    #zz_steps = [0.009]
    perturbDataPct = 0.0002
    longMemory =  False
    iterations=1
    input_signal = 1
    feature_selection = 'None' #RFECV OR Univariate
    wfSteps=[1]
    wf_is_periods = [500,1000]
    #wf_is_periods = [10]
    tox_adj_proportion = 0
    nfeatures = 10

    #feature Parameters
    RSILookback = 1.5
    zScoreLookback = 10
    ATRLookback = 5
    beLongThreshold = 0.0
    DPOLookback = 10
    ACLookback = 12
    CCLookback = 60
    rStochLookback = 300
    statsLookback = 100
    ROCLookback = 600

    #DPS parameters
    #windowLengths = [2] 
    windowLengths = [30]
    maxLeverage = [5]
    PRT={}
    PRT['DD95_limit'] = 0.01
    PRT['tailRiskPct'] = 95
    PRT['initial_equity'] = 1.0
    PRT['horizon'] = 720
    PRT['maxLeverage'] = 5
    #CAR25_threshold=-np.inf
    CAR25_threshold=0

    #model selection
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    RFE_estimator = [ 
            ("None","None"),\
            #("GradientBoostingRegressor",GradientBoostingRegressor()),\
            #("DecisionTreeRegressor",DecisionTreeRegressor()),\
            #("ExtraTreeRegressor",ExtraTreeRegressor()),\
            #("BayesianRidge", BayesianRidge()),\
             ]
             
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
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             #("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:1}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             ("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 ("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                    ], voting='hard', weights=None)),
             #("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #        ], voting='soft', weights=None)),
             ]    


    ############################################################
    currencyPairsDict = {}
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

    for pair in currencyPairs:    
        if barSizeSetting+'_'+pair+'.csv' in files:
            data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)
            currencyPairsDict[pair] = data
            
    print 'Successfully Retrieved Data.'
    ###########################################################
    print 'Begin Preprocessing...'

    #add cross pair features
    print 'removing dupes..'
    currencyPairsDict2 = copy.deepcopy(currencyPairsDict)
    for pair in currencyPairsDict2:
        data = currencyPairsDict2[pair].copy(deep=True)
        #perturb dataSet
        if perturbData:
            print 'perturbing OHLC and',
            data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
            data['High']= perturb_data(data['High'].values,perturbDataPct)
            data['Low']= perturb_data(data['Low'].values,perturbDataPct)
            data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
            
        
        print pair, currencyPairsDict2[pair].shape,'to',
        currencyPairsDict2[pair] = data.drop_duplicates()
        print pair, currencyPairsDict2[pair].shape

    #if last index exists in all pairs append to dataSet
    dataSet = currencyPairsDict2[ticker].copy(deep=True)

    for pair in currencyPairsDict2:
        if dataSet.shape[0] != currencyPairsDict2[pair].shape[0]:
            print 'Warning:',pair, 'row mismatch. Some Data may be lost.'
            #print 'Adjusting rows in dataSet and cross-pair to match', ticker, dataSet.shape[0],\
            #            pair, currencyPairsDict2[pair].shape[0], 'to',
            #intersect =np.intersect1d(currencyPairsDict2[pair].index, dataSet.index)         
            #dataSet = dataSet.ix[intersect]:
            #currencyPairsDict2[pair] = currencyPairsDict2[pair].ix[intersect]
            #print ticker, dataSet.shape[0], pair, currencyPairsDict2[pair].shape[0]
                            
    nrows = dataSet.shape[0]
    lastIndex = currencyPairsDict2[ticker].index[-1]
    for pair in currencyPairsDict2:
        if lastIndex in currencyPairsDict2[pair].index and pair != ticker and\
                            (ticker[0:3] in pair or ticker[3:6] in pair):
            intersect =np.intersect1d(currencyPairsDict2[pair].index, dataSet.index)         
            dataSet = dataSet.ix[intersect]
            currencyPairsDict2[pair] = currencyPairsDict2[pair].ix[intersect]
            
            closes = pd.concat([dataSet.Close, currencyPairsDict2[pair].Close],\
                                    axis=1, join='inner')
                                                           
            dataSet['corr'+pair] = pd.rolling_corr(closes.iloc[:,0],\
                                            closes.iloc[:,1], window=CCLookback)
            dataSet['priceChange'+pair] = priceChange(closes.iloc[:,1])
            dataSet['Pri_rStoch'+pair] = roofingFilter(closes.iloc[:,1],rStochLookback,500)
            dataSet['ROC_'+pair] = ROC(closes.iloc[:,1],ROCLookback)
    
    print nrows-dataSet.shape[0], 'rows lost for', ticker
    
    #short indicators
    dataSet['Pri_RSI'] = RSI(dataSet.Close,RSILookback)
    dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
    dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
    dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
    dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)

    #long indicators
    dataSet['Pri_ROC'] = ROC(dataSet.Close,ROCLookback)
    dataSet['Pri_rStoch'] = roofingFilter(dataSet.Close,rStochLookback,500)
    dataSet['Pri_rStoch_Y1'] = dataSet['Pri_rStoch'].shift(1)
    dataSet['Pri_rStoch_Y2'] = dataSet['Pri_rStoch'].shift(2)
    dataSet['Pri_rStoch_Y3'] = dataSet['Pri_rStoch'].shift(3)
    dataSet['Pri_rStoch_Y4'] = dataSet['Pri_rStoch'].shift(4)

    #volatility
    dataSet['Pri_ATR'] = zScore(ATR(dataSet.High,dataSet.Low,dataSet.Close,ATRLookback),
                              zScoreLookback)
    dataSet['Pri_ATR_Y1'] = dataSet['Pri_ATR'].shift(1)
    dataSet['Pri_ATR_Y2'] = dataSet['Pri_ATR'].shift(2)
    dataSet['Pri_ATR_Y3'] = dataSet['Pri_ATR'].shift(3)
    dataSet['Pri_ATR_Y4'] = dataSet['Pri_ATR'].shift(4)
    dataSet['priceChange'] = priceChange(dataSet.Close)
    dataSet['priceChangeY1'] = dataSet['priceChange'].shift(1)
    dataSet['priceChangeY2'] = dataSet['priceChange'].shift(2)
    dataSet['priceChangeY3'] = dataSet['priceChange'].shift(3)
    dataSet['priceChangeY4'] = dataSet['priceChange'].shift(4)
    dataSet['mean60_ga'] = pd.rolling_mean(dataSet.priceChange,statsLookback)
    dataSet['std60_ga'] = pd.rolling_std(dataSet.priceChange,statsLookback)

    #correlations
    dataSet['autoCor'] = autocorrel(dataSet.Close*100,ACLookback)
    dataSet['autoCor_Y1'] = dataSet['autoCor'].shift(1)
    dataSet['autoCor_Y2'] = dataSet['autoCor'].shift(2)
    dataSet['autoCor_Y3'] = dataSet['autoCor'].shift(3)
    dataSet['autoCor_Y4'] = dataSet['autoCor'].shift(4)

    #labels
    dataSet['gainAhead'] = gainAhead(dataSet.Close)
    dataSet['signal'] =  np.where(dataSet.gainAhead>0,1,-1)
    
    print 'Creating Signal labels..'
    if 'ZZ' in signal_types:
        signal_types.remove('ZZ')
        for i in zz_steps:
            #for j in zz_steps:
            label = 'ZZ '+str(i) + ',-' + str(i)
            print label+',',
            signal_types.append(label)
            #zz_signals[label] = zg(dataSet.Close, i, -j).pivots_to_modes()
            
    if 'gainAhead' in signal_types:
        signal_types.remove('gainAhead')
        for i in ga_steps:
            label = 'GA'+str(i)
            print label+',',
            signal_types.append(label)
            
    #find max lookback
    maxlb = max(RSILookback,
                        zScoreLookback,
                        ATRLookback,
                        DPOLookback,
                        ACLookback,
                        rStochLookback,
                        ROCLookback,
                        CCLookback)
    # add shift
    maxlb = maxlb+4

    #raise error if nan/inf in dataSet
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

    dataSet = dataSet.ix[maxlb:].dropna()
    dataSet['prior_index'] = dataSet.dropna().reset_index().index
    dataSet.index = dataSet.index.values.astype(str)
    dataSet.index = dataSet.index.to_datetime()
    dataSet.index.name ='dates'
    unfilteredData = dataSet.copy(deep=True)
    
    if dataSet.shape[0] <validationSetLength+max(wf_is_periods):
        message = 'Add more data: dataSet rows '+str(dataSet.shape[0])+\
                    ' is less than required validation set + training set of '\
                    + str(validationSetLength+max(wf_is_periods))
        offlineMode(ticker, message, signalPath, version, version_)
    else:
        for i in validationPeriods:
            if dataSet.iloc[-validationSetLength:].shape[0]%i != 0:
                message='validationSetLength '+str(validationSetLength)+\
                        ' needs to be divisible by validation period '+ str(i)
                offlineMode(ticker, message, signalPath, version, version_)
            if validationSetLength == i:
                message='validationSetLength '+str(validationSetLength)+\
                        ' needs to be less than validation period '+ str(i)
                offlineMode(ticker, message, signalPath, version, version_)
                
        dataSet = dataSet.iloc[-validationSetLength-max(wf_is_periods):]
        print 'dataSet rows', dataSet.shape[0], 'from', dataSet.index[0], 'to', dataSet.index[-1]

        

    print '\n\nNew simulation run... '
    if showDist:
        describeDistribution(dataSet.reset_index()['Close'], dataSet.reset_index()['priceChange'], ticker)

    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Pre-processing... Beginning Model Training..'

    #########################################################
    validationDict = {}
    BMdict = {}
    for validationPeriod in validationPeriods:
        #DPScycle = 0
        endOfData = 0
        t_start_loc = max(wf_is_periods)
        if validationPeriod > dataSet.shape[0]+t_start_loc:
            print 'Validation Period', validationPeriod, '> dataSet + training period',
            validationPeriod = dataSet.shape[0]-t_start_loc-1
            print 'adjusting to', validationPeriod, 'rows'
        elif validationPeriod <2:
            print 'Validation Period', validationPeriod, '< 2',
            validationPeriod = 2
            print 'adjusting to 2 rows'
        
        testFirstYear = dataSet.index[0]
        testFinalYear = dataSet.index[t_start_loc-1]
        validationFirstYear =dataSet.index[t_start_loc]   
        validationFinalYear =dataSet.index[t_start_loc+validationPeriod-1]
        
        #check if validation start point is less than available training data
        #if validationStartPoint is not None and\
        #            dataSet.shape[0]>dataSet.index[-validationStartPoint:].shape[0]+t_start_loc:
        #    testFinalYear = dataSet.index[-validationStartPoint-1]
        #    validationFirstYear =dataSet.index[-validationStartPoint]
            
        while endOfData<2:
            #init
            model_metrics = init_report()       
            sstDictDF1_ = {} 
            #DPScycle+=1
            
            for signal in signal_types:
                for m in models:
                    for wfStep in wfSteps:
                        for wf_is_period in wf_is_periods:

                            #check
                            nrows_is = dataSet.ix[:testFinalYear].shape[0]
                            if wf_is_period > nrows_is:
                                print 'Walkforward insample period of', wf_is_period,\
                                        'is greater than in-sample data of ', nrows_is, '!'
                                print 'Adjusting to', nrows_is, 'rows..'
                                wf_is_period = nrows_is
      
                            metaData = {'ticker':ticker, 't_start':testFirstYear, 't_end':testFinalYear,\
                                    'signal':signal, 'data_type':data_type,'filter':filterName, 'input_signal':input_signal,\
                                    'test_split':0, 'iters':1, 'tox_adj':tox_adj_proportion,'longMemory':longMemory,\
                                    'n_features':nfeatures, 'FS':feature_selection,'rfe_model':RFE_estimator[0],\
                                    'v_start':validationFirstYear, 'v_end':validationFinalYear,'wf_step':wfStep,\
                                    'RSILookback': RSILookback,'zScoreLookback': zScoreLookback,'ATRLookback': ATRLookback,\
                                    'beLongThreshold': beLongThreshold,'DPOLookback': DPOLookback,'ACLookback': ACLookback,\
                                    'CCLookback': CCLookback,'rStochLookback': rStochLookback,'statsLookback': statsLookback,\
                                    'ROCLookback': ROCLookback,
                                     }
                            runName = ticker+'_'+data_type+'_'+filterName+'_' + m[0]+'_i'+str(wf_is_period)+'_fcst'+str(wfStep)+'_'+signal
                            model_metrics, sstDictDF1_[runName] = wf_classify_validate(unfilteredData, dataSet, [m], model_metrics,\
                                                                wf_is_period, metaData, PRT, showPDFCDF=showPDFCDF)

            #score models
            scored_models, bestModel = directional_scoring(model_metrics,filterName)
            #bestModel['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            #bestModel = bestModel.append(pd.Series(data=datetime.datetime.fromtimestamp(time.time())\
            #                               .strftime('%Y-%m-%d %H:%M:%S'), index=['timestamp']))
                                            
            #keep original for other DF
            #sstDictDF1_Combined_DF1_beShorts_ = copy.deepcopy(sstDictDF1_)
            #sstDictDF1_DF1_Shorts_beFlat_ = copy.deepcopy(sstDictDF1_)
            if showAllCharts:
                for runName in sstDictDF1_:
                    compareEquity(sstDictDF1_[runName],runName)
                
            for m in models:
                if bestModel['params'] == str(m[1]):
                    print  '\n\nBest model found...\n', m[1]
                    bm = m[1]
            print 'Number of features: ', bestModel.n_features, bestModel.FS
            print 'WF In-Sample Period:', bestModel.rows
            print 'WF Out-of-Sample Period:', bestModel.wf_step
            print 'Long Memory: ', longMemory
            DF1_BMrunName = ticker+'_'+bestModel.data_type+'_'+filterName+'_'  +\
                                bestModel.model + '_i'+str(bestModel.rows)\
                                +'_fcst'+str(bestModel.wf_step)+'_'+bestModel.signal
                
            if showAllCharts:
                compareEquity(sstDictDF1_[DF1_BMrunName],DF1_BMrunName)
                
            print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
            print 'Finished Model Training...'
            
            if runDPS:               
                print 'Beginning Dynamic Position Sizing..'
                for wl in windowLengths:
                    if validationPeriod in validationDict and \
                           len([x for x in validationDict[validationPeriod].index if x not in sstDictDF1_[DF1_BMrunName].index]) >wl:
                    #prior validation data available from the third DPScycle. wl<1st cycle vp
                        sst_bestModel = validationDict[validationPeriod].drop(['safef','CAR25','dd95','ddTol','system'], axis=1)
                        zero_index = np.array([x for x in sst_bestModel.index if x not in sstDictDF1_[DF1_BMrunName].index])[-int(wl):]
                        sst_zero = sst_bestModel.ix[zero_index]
                        sst_bestModel = pd.concat([sst_zero,sstDictDF1_[DF1_BMrunName]],axis=0)
                        
                        #adjust window length if it is larger than zero index
                        if sst_zero.shape[0]<wl:
                            wl=sst_zero.shape[0]
                    else:
                        #add zeros if two cycles
                        zero_index = dataSet.ix[:sstDictDF1_[DF1_BMrunName].index[0]].index[:-1]
                        sst_zero = pd.concat([pd.Series(data=0, name='signals', index=zero_index ), dataSet.gainAhead.ix[zero_index],\
                                                        dataSet.prior_index.ix[zero_index]], axis=1)
                        sst_bestModel = pd.concat([sst_zero,sstDictDF1_[DF1_BMrunName]],axis=0)
                        
                        #adjust window length if it is larger than zero index
                        if sst_zero.shape[0]<wl:
                            wl=sst_zero.shape[0]

                    #calc DPS
                    #DPS = {}
                    DPS_both = {}

                    startDate = sstDictDF1_[DF1_BMrunName].index[0]
                    endDate = sst_bestModel.index[-1]
                    for ml in maxLeverage:
                        PRT['maxLeverage'] = ml               
                        dpsRun, sst_save = calcDPS2(DF1_BMrunName, sst_bestModel, PRT, startDate,\
                                                                    endDate, wl, 'both', threshold=CAR25_threshold)
                        DPS_both[dpsRun] = sst_save
                        
                        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                        #DPS[dpsRun] = sst_save
                    
                dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                                            endDate,'both', DF1_BMrunName, yscale='linear',\
                                                            ticker=ticker,displayCharts=showAllCharts,\
                                                            equityStatsSavePath=equityStatsSavePath)
                bestBothDPS.index.name = 'dates'
                bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system')], \
                                                                axis=1)
                if endOfData ==1:
                    #save sst's for save file         
                    BMdict[validationPeriod] = bestBothDPS
                    #save model
                    
                    
                if showAllCharts:
                    DPSequity = calcEquity_df(bestBothDPS[['signals','gainAhead']], dpsRunName, leverage = bestBothDPS.safef.values)
            else:
                BMdict[validationPeriod] = pd.concat([sstDictDF1_[DF1_BMrunName],\
                                    pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=DF1_BMrunName, name = 'system', index = sstDictDF1_[DF1_BMrunName].index)
                                    ],axis=1)
                
            #use best params for next step
            for i in range(0,bestModel.shape[0]):
                metaData[bestModel.index[i]]=bestModel[bestModel.index[i]]
            
            #adjust test dates
            t_start_loc = dataSet.reset_index()[dataSet.reset_index().dates ==bestModel.v_end].index[0]
            testFirstYear = dataSet.index[0]
            testFinalYear = dataSet.index[t_start_loc]
            
            #if t_start_loc+1 >= dataSet.shape[0]:
            #    validationFirstYear =dataSet.index[t_start_loc]
            #else:
            if endOfData<1:
                validationFirstYear = dataSet.index[t_start_loc+1]
                validationFinalYear =dataSet.index[t_start_loc+validationPeriod]
            
            #if t_start_loc+validationPeriod >= dataSet.shape[0]:
            if validationFinalYear == dataSet.index[-1]:
                endOfData+=1
            #else:
            
            
            
            if endOfData<2:
                print '\n\nBest Params Found... Simulating Next Period..'
                #update metaData dates
                metaData['t_start']=testFirstYear
                metaData['t_end']=testFinalYear
                metaData['v_start']=validationFirstYear
                metaData['v_end']=validationFinalYear
                nextRunName = 'Next'+str(validationPeriod)+'_'+DF1_BMrunName
                model_metrics, sstDictDF1_[nextRunName] = wf_classify_validate(unfilteredData, dataSet, [m], model_metrics,\
                                                    wf_is_period, metaData, PRT, showPDFCDF=showPDFCDF)
                if showAllCharts:
                    compareEquity(sstDictDF1_[nextRunName],nextRunName)
                print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
                print 'Finished Next Period Model Training... '   
                if runDPS:
                    #if DPS was chosen as the best model to use going forward
                    if dpsRunName.find('wl') != -1:
                        #for ml in maxLeverage:
                        PRT['maxLeverage'] = float(dpsRunName[dpsRunName.find('wl'):].split()[1][4:])
                        #for wl in windowLengths:
                        wl=float(dpsRunName[dpsRunName.find('wl'):].split()[0][2:])
                        
                        print 'Beginning Next Period Dynamic Position Sizing..'
                        if validationPeriod in validationDict and validationDict[validationPeriod].shape>=wl:
                            sst_bestModel = pd.concat([validationDict[validationPeriod].drop(['safef','CAR25','dd95','ddTol','system'],axis=1),\
                                                                sstDictDF1_[nextRunName]], axis=0)
                        else:
                            #sst_bestModel = pd.concat([sstDictDF1_[DF1_BMrunName],\
                            #                                   sstDictDF1_[nextRunName]], axis=0)
                                                                
                            #add zeros if two cycles
                            zero_index = np.array([x for x in sstDictDF1_[DF1_BMrunName].index\
                                                                if x not in sstDictDF1_[nextRunName].index])[-int(wl):]
                            sst_zero = pd.concat([pd.Series(data=0, name='signals', index=zero_index ),\
                                                            sstDictDF1_[DF1_BMrunName].gainAhead.ix[zero_index],\
                                                            sstDictDF1_[DF1_BMrunName].prior_index.ix[zero_index]], axis=1)
                            sst_bestModel = pd.concat([sst_zero,sstDictDF1_[nextRunName]],axis=0)
                            
                            #adjust window length if it is larger than zero index
                            if sst_zero.shape[0]<wl:
                                wl=sst_zero.shape[0]
                            
                        #calc DPS
                        DPS_both = {}

                        startDate = sstDictDF1_[nextRunName].index[0]
                        endDate = sst_bestModel.index[-1]
                        dpsRun, sst_save = calcDPS2(nextRunName, sst_bestModel, PRT, startDate,\
                                                                    endDate, wl, 'both', threshold=CAR25_threshold)
                        DPS_both[dpsRun] = sst_save
                        
                        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                        #DPS[dpsRun] = sst_save
                            
                        dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                                                    endDate,'both', nextRunName, yscale='linear',\
                                                                    ticker=ticker,displayCharts=showAllCharts,\
                                                                    equityStatsSavePath=equityStatsSavePath)
                        bestBothDPS.index.name = 'dates'
                        bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system'),\
                                                                ], axis=1)
                        if showAllCharts:                                       
                            DPSequity = calcEquity_df(bestBothDPS[['signals','gainAhead']], dpsRunName, leverage = bestBothDPS.safef.values)
                        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
                        print 'Finished Dynamic Position Sizing..'
                        
                        if validationPeriod not in validationDict:
                            validationDict[validationPeriod] =bestBothDPS
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                    message = "Error: NaN in bestBothDPS.prior_index:1 "
                                    offlineMode(ticker, message, signalPath, version, version_)
                        else:              
                            validationDict[validationPeriod] =validationDict[validationPeriod].append(bestBothDPS)
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                    message="Error: NaN in bestBothDPS.prior_index:2 "
                                    offlineMode(ticker, message, signalPath, version, version_)
                    else:
                        #if No DPS was chosen as the best model to use going forward
                        if validationPeriod not in validationDict:
                            validationDict[validationPeriod] =    pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                message="Error: NaN in bestBothDPS.prior_index:3 "
                                offlineMode(ticker, message, signalPath, version, version_)    
                        else:                      
                            validationDict[validationPeriod] =validationDict[validationPeriod].append(                                                                         
                                                                              pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)                                                               
                                                                             )
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                message="Error: NaN in bestBothDPS.prior_index:4"
                                offlineMode(ticker, message, signalPath, version, version_)    
                               
                else:   
                    #DPSrun == False
                    if validationPeriod not in validationDict:
                        validationDict[validationPeriod] =pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)
                    else:
                        validationDict[validationPeriod] =validationDict[validationPeriod].append(                                                                         
                                                                              pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)                                                               
                                                                             )
          
    print '\n\nScoring Validation Curves...'
    #set validation curves equal length
    vStartDate = dataSet.index[0]
    for validationPeriod in validationDict:
        if validationDict[validationPeriod].index[0]>vStartDate:
            vStartDate =  validationDict[validationPeriod].index[0]
    
    if runDPS:
        vCurve_metrics = init_report()
        for validationPeriod in validationDict:
            vCurve = ticker+' Validation Period '+str(validationPeriod)
            #compareEquity(validationDict[validationPeriod],vCurve)
            if showAllCharts:
                DPSequity = calcEquity_df(validationDict[validationPeriod].ix[vStartDate:][['signals','gainAhead']], vCurve,\
                                                    leverage = validationDict[validationPeriod].ix[vStartDate:].safef.values)
            if scorePath is not None:
                validationDict[validationPeriod].to_csv(equityStatsSavePath+vCurve+'_'+\
                                                                str(validationDict[validationPeriod].index[0]).replace(':','')+'_to_'+\
                                                                str(validationDict[validationPeriod].index[-1]).replace(':','')+'.csv')
            CAR25_oos = CAR25_df_min(vCurve,validationDict[validationPeriod].ix[vStartDate:].signals,\
                                                validationDict[validationPeriod].ix[vStartDate:].prior_index.values.astype(int),\
                                                unfilteredData.Close, minFcst=PRT['horizon'] , DD95_limit =PRT['DD95_limit'] )
            model = [validationDict[validationPeriod].iloc[-1].system.split('_')[4],\
                            [m[1] for m in models if m[0] ==validationDict[validationPeriod].iloc[-1].system.split('_')[4]][0]]
            metaData['validationPeriod'] = validationPeriod
            #metaData['params'] =model[1]
            vCurve_metrics = update_report(vCurve_metrics, filterName,\
                                                validationDict[validationPeriod].ix[vStartDate:].signals.values.astype(int), \
                                                dataSet.ix[validationDict[validationPeriod].ix[vStartDate:].index].signal.values.astype(int),\
                                                unfilteredData.gainAhead,\
                                                validationDict[validationPeriod].ix[vStartDate:].prior_index.values.astype(int), model,\
                                                metaData,CAR25_oos)
    else:
        vCurve_metrics = init_report()
        for validationPeriod in validationDict:
            vCurve = ticker+' Validation Period '+str(validationPeriod)
            if showAllCharts:
                compareEquity(validationDict[validationPeriod].ix[vStartDate:],vCurve)
            if scorePath is not None:
                validationDict[validationPeriod].to_csv(equityStatsSavePath+vCurve+'_'+\
                                                                str(validationDict[validationPeriod].index[0]).replace(':','')+'_to_'+\
                                                                str(validationDict[validationPeriod].index[-1]).replace(':','')+'.csv')
            CAR25_oos = CAR25_df_min(vCurve,validationDict[validationPeriod].ix[vStartDate:].signals,\
                                                validationDict[validationPeriod].ix[vStartDate:].prior_index.values.astype(int),\
                                                unfilteredData.Close, minFcst=PRT['horizon'] , DD95_limit =PRT['DD95_limit'] )
            model = [validationDict[validationPeriod].iloc[-1].system.split('_')[4],\
                            [m[1] for m in models if m[0] ==validationDict[validationPeriod].iloc[-1].system.split('_')[4]][0]]
            #metaData['model'] = model[0]
            #metaData['params'] =model[1]
            metaData['validationPeriod'] = validationPeriod
            vCurve_metrics = update_report(vCurve_metrics, filterName,\
                                                validationDict[validationPeriod].ix[vStartDate:].signals.values.astype(int), \
                                                dataSet.ix[validationDict[validationPeriod].ix[vStartDate:].index].signal.values.astype(int),\
                                                unfilteredData.gainAhead,\
                                                validationDict[validationPeriod].ix[vStartDate:].prior_index.values.astype(int), model,\
                                                metaData,CAR25_oos)
            
    #score models
    scored_models, bestModel = directional_scoring(vCurve_metrics,filterName)
    #bestModel['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    
    print '\n\nBest Validation Period Found:', bestModel.validationPeriod
    if showAllCharts:
        BestEquity = calcEquity_df(validationDict[bestModel.validationPeriod][['signals','gainAhead']],\
                                            bestModel.C25sig, leverage = validationDict[bestModel.validationPeriod].safef.values)
    timenow = dt.now(timezone('US/Eastern'))
    lastBartime = timezone('US/Eastern').localize(dataSet.index[-1].to_datetime())
    cycleTime = (timenow-lastBartime).total_seconds()/60
    bestModel = bestModel.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
    bestModel = bestModel.append(pd.Series(data=lastBartime.strftime("%Y%m%d %H:%M:%S %Z"), index=['lastBarTime']))
    bestModel = bestModel.append(pd.Series(data=cycleTime, index=['cycleTime']))
    
    print version, 'Saving Params..'
    files = [ f for f in listdir(bestParamsPath) if isfile(join(bestParamsPath,f)) ]
    if version+'_'+ticker + '.csv' not in files:
        BMdf = pd.concat([bestModel,bestModel],axis=1).transpose()
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'.csv', index=False)
    else:
        BMdf = pd.read_csv(bestParamsPath+version+'_'+ticker+'.csv').append(bestModel, ignore_index=True)
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'.csv', index=False)
    
    if scorePath is not None:
        scored_models.to_csv(scorePath+version+'_'+ticker+'.csv')
        
    #if runDPS:    
    print '\n'+version, 'Next Signal:'
    print BMdict[bestModel.validationPeriod].iloc[-1].system
    print BMdict[bestModel.validationPeriod].drop(['system'],axis=1).iloc[-1]
     
    print 'Saving Signals..'      
    #init file
    #BMdict[bestModel.validationPeriod].tail().to_csv(signalPath + version+'_'+ ticker + '.csv')
    files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
    
    #old version file
    if version+'_'+ ticker + '.csv' not in files:
        signalFile = BMdict[bestModel.validationPeriod].tail()
        signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker + '.csv', parse_dates=['dates'])

        #if BMdict[bestModel.validationPeriod].reset_index().iloc[-1].dates != signalFile.iloc[-1].dates:
        if BMdict[bestModel.validationPeriod].reset_index().iloc[-2].dates == signalFile.iloc[-1].dates:
            signalFile.gainAhead.iloc[-1] == BMdict[bestModel.validationPeriod].gainAhead.iloc[-2]
        signalFile=signalFile.append(BMdict[bestModel.validationPeriod].reset_index().iloc[-1])
        signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=False)
        
    #new version file
    if version_+'_'+ ticker + '.csv' not in files:
        #signalFile = BMdict[bestModel.validationPeriod].iloc[-2:]
        addLine = BMdict[bestModel.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = BMdict[bestModel.validationPeriod].iloc[-1].name
        signalFile = BMdict[bestModel.validationPeriod].iloc[-2:-1].append(addLine)
        signalFile.index.name = 'dates'
        signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker + '.csv', index_col=['dates'])
        addLine = BMdict[bestModel.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = BMdict[bestModel.validationPeriod].iloc[-1].name
        signalFile = signalFile.append(addLine)
        signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
    
            
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Validation Period Parameter Search!'
    
    ##############################################################
