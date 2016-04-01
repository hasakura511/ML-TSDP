
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015
changelog
v3.1
model drop for v1/v2
added wf stepping for ga and zz signals
added verbose mode
added buyhold/sellhold signal types
dps mode/nodps modes
chart saving
bugfixes

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
                         offlineMode, oos_display_cmatrix
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df_bars,\
                            maxCAR25, wf_classify_validate2, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS,\
                            calcEquity_df
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter, getCycleTime
                        
from suztoolz.transform import zigzag as zg
from suztoolz.data import getDataFromIB

start_time = time.time()

#system parameters
version = 'v3'
version_ = 'v3.1'

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='30m'
#barSizeSetting='1 min'
currencyPairs = ['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']


    
def saveParams(dataSet, bestModelParams, bestParamsPath, modelSavePath, version, barSizeSetting):
    timenow, lastBartime, cycleTime = getCycleTime(dataSet)
    bestModelParams = bestModelParams.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
    bestModelParams = bestModelParams.append(pd.Series(data=lastBartime.strftime("%Y%m%d %H:%M:%S %Z"), index=['lastBartime']))
    bestModelParams = bestModelParams.append(pd.Series(data=cycleTime, index=['cycleTime']))
    
    print '\n'+'Saving Params for '+version
    files = [ f for f in listdir(bestParamsPath) if isfile(join(bestParamsPath,f)) ]
    if version+'_'+ticker+'_'+barSizeSetting+ '.csv' not in files:
        BMdf = pd.concat([bestModelParams,bestModelParams],axis=1).transpose()
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv', index=False)
    else:
        BMdf = pd.read_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv').append(bestModelParams, ignore_index=True)
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv', index=False)
    
    print 'Saving Model for '+version
    joblib.dump(BSMdict[bestModelParams.validationPeriod], modelSavePath+version+'_'+ticker+'_'+barSizeSetting+'.joblib',compress=3)

        
        

        
#no args -> debug.  else live mode arg 1 = pair, arg 2 = "0" to turn off
if len(sys.argv)==1:
    debug=True
    
    livePairs =  [
                    #'NZDJPY',\
                    #'CADJPY',\
                    #'CHFJPY',\
                    #'EURJPY',\
                    #'GBPJPY',\
                    #'AUDJPY',\
                    #'USDJPY',\
                    #'AUDUSD',\
                    #'EURUSD',\
                    #'GBPUSD',\
                    #'USDCAD',\
                    'USDCHF',\
                    #'NZDUSD',
                    #'EURCHF',\
                    #'EURGBP'\
                    ]
                    
                    
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    showBestCharts = True
    perturbData = True
    runDPS = True
    verbose= False
    returnNoDPS = False
    #scorePath = './debug/scored_metrics_'
    #equityStatsSavePath = './debug/'
    #signalPath = './debug/'
    #dataPath = './data/from_IB/'
    scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    modelSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/models/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/' 
    
else:
    print 'Live Mode', sys.argv[1], sys.argv[2]
    debug=False
    
    #display
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    showBestCharts = False
    perturbData = True
    runDPS = True
    verbose= False
    returnNoDPS=False
    
    #paths
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    bestParamsPath =  './data/params/'
    modelSavePath = './data/models/' 
    chartSavePath = './data/results/' 
    
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
    maxReadLines = 5000
    #dataSet length needs to be divisiable by each validation period! 
    validationSetLength = 360
    #validationSetLength = 1200
    #validationPeriods = [50]
    validationPeriods = [30,120] # min is 2
    #validationStartPoint = None
    #signal_types = ['buyHold','sellHold']
    #signal_types = ['gainAhead','buyHold','sellHold']
    signal_types = ['gainAhead','zigZag','buyHold','sellHold']
    #signal_types = ['zigZag']
    #signal_types = ['gainAhead']
    zz_steps = [0.002]
    #zz_steps = [0.009]
    #wfSteps=[1,30,60]
    wfSteps=[1,15,30]
    wf_is_periods = [250,500]
    #wf_is_periods = [250,500,1000]
    perturbDataPct = 0.0002
    longMemory =  False
    iterations=1
    input_signal = 1
    feature_selection = 'None' # OR Univariate
    #feature_selection = 'RFECV'
    #feature_selection = 'Univariate'
    nfeatures = 10
    tox_adj_proportion = 0
    
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
    windowLengths = [10,50]
    maxLeverage = [2]
    PRT={}
    PRT['initial_equity'] = 1
    #fcst horizon(bars) for dps.  for training,  horizon is set to nrows.  for validation scoring nrows. 
    PRT['horizon'] = 50
    #safef set to dd95 where limit is met. e.g. for 50 bars, set saef to where 95% of the mc eq curves' maxDD <=0.01
    PRT['DD95_limit'] = 0.01
    PRT['tailRiskPct'] = 95
    #this will be reset later
    PRT['maxLeverage'] = 5
    #safef=0 if CAR25 < threshold
    PRT['CAR25_threshold'] = 0
    #PRT['CAR25_threshold'] = -np.inf


    #model selection
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    RFE_estimator = [ 
            ("None","None"),\
            #("GradientBoostingRegressor",GradientBoostingRegressor()),\
            #("DecisionTreeRegressor",DecisionTreeRegressor()),\
            #("ExtraTreeRegressor",ExtraTreeRegressor()),\
            #("BayesianRidge", BayesianRidge()),\
             ]
             
    models = [('None','None'),
              #("GA_Reg", SymbolicRegressor(population_size=5000, generations=20,
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
            if data.shape[0] >= maxReadLines:
                currencyPairsDict[pair] = data[-maxReadLines:]
            else:
                currencyPairsDict[pair] = data
            
    print 'Successfully Retrieved Data. Beginning Preprocessing...'
    ###########################################################
    #print 'Begin Preprocessing...'

    #add cross pair features
    #print 'removing dupes..'
    currencyPairsDict2 = copy.deepcopy(currencyPairsDict)
    for pair in currencyPairsDict2:
        data = currencyPairsDict2[pair].copy(deep=True)
        #perturb dataSet
        if perturbData:
            if verbose:
                print 'perturbing OHLC and dropping dupes',
            data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
            data['High']= perturb_data(data['High'].values,perturbDataPct)
            data['Low']= perturb_data(data['Low'].values,perturbDataPct)
            data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
            
        if verbose:
            print pair, currencyPairsDict2[pair].shape,'to',
        currencyPairsDict2[pair] = data.drop_duplicates()
        if verbose:
            print pair, currencyPairsDict2[pair].shape

    #if last index exists in all pairs append to dataSet
    dataSet = currencyPairsDict2[ticker].copy(deep=True)

    for pair in currencyPairsDict2:
        if dataSet.shape[0] != currencyPairsDict2[pair].shape[0]:
            if verbose:
                print 'Warning:',ticker, pair, 'row mismatch. Some Data may be lost.'
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
    
    #check all wfStep < max(validationPeriods)
    wfSteps=[step for step in wfSteps if step <= max(validationPeriods)]
        
    print '\nCreating Signal labels..'
    #for wfStep in wfSteps:   
    #    print 'Step', wfStep,
    if 'zigZag' in signal_types:
        signal_types.remove('zigZag')
        for wfStep in wfSteps:   
            for i in zz_steps:
                #for j in zz_steps:
                label = 'ZZ'+str(wfStep)+'_'+str(i) + ',-' + str(i)
                print label+',',
                signal_types.append(label)
                #zz_signals[label] = zg(dataSet.Close, i, -j).pivots_to_modes()
                
    if 'gainAhead' in signal_types:
        signal_types.remove('gainAhead')
        for wfStep in wfSteps:   
            label = 'GA'+str(wfStep)
            print label+',',
            signal_types.append(label)
            
    for wfStep in wfSteps:   
        if 'buyHold' in signal_types:
            signal_types.remove('buyHold')
            label = 'BH'+str(wfStep)
            print label+',',
            signal_types.append(label)

        if 'sellHold' in signal_types:
            signal_types.remove('sellHold')
            label = 'SH'+str(wfStep)
            print label+',',
            signal_types.append(label)           
    print '\nCreated',len(signal_types),'signal types to check'
            
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
    
    if dataSet.shape[0] < validationSetLength+max(wf_is_periods)+max(validationPeriods):
        message = 'Add more data: dataSet rows '+str(dataSet.shape[0])+\
                    ' is less than required validation set + first optimisation period + training set of '\
                    + str(validationSetLength+max(wf_is_periods)+max(validationPeriods))
        offlineMode(ticker, message, signalPath, version, version_)
    else:
        for i in validationPeriods:
            if dataSet.iloc[-validationSetLength:].shape[0]%i != 0:
                message='validationSetLength '+str(validationSetLength)+\
                        ' needs to be divisible by validation period '+ str(i)
                offlineMode(ticker, message, signalPath, version, version_)
            if i <2:
                message='Validation Period ' + str(validationPeriod)+ ' needs to be greater than 2'
                offlineMode(ticker, message, signalPath, version, version_)
            #if validationSetLength == i:
            #    message='validationSetLength '+str(validationSetLength)+\
            #            ' needs to be less than validation period '+ str(i)
            #    offlineMode(ticker, message, signalPath, version, version_)
                
        dataSet = dataSet.iloc[-validationSetLength-max(wf_is_periods)-max(validationPeriods):]
        print '\nProcessed ', dataSet.shape[0], 'rows from', dataSet.index[0], 'to', dataSet.index[-1]

        
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    
    if showDist:
        describeDistribution(dataSet.reset_index()['Close'], dataSet.reset_index()['priceChange'], ticker)  
    #print 'Finished Pre-processing... Beginning Model Training..'

    #########################################################
    validationDict_noDPS = {}
    validationDict_DPS = {}
    v3signals = {}
    BSMdict={}
    for validationPeriod in validationPeriods:
        print '\nStarting optimization run for validation period of',validationPeriod
        #DPScycle = 0
        endOfData = 0
        t_start_loc = max(wf_is_periods)+max(validationPeriods)-validationPeriod
        #if validationPeriod > dataSet.shape[0]+t_start_loc:
        #    print 'Validation Period', validationPeriod, '> dataSet + training period',
        #    validationPeriod = dataSet.shape[0]-t_start_loc-1
        #    print 'adjusting to', validationPeriod, 'rows'
        #elif validationPeriod <2:
        #    print 'Validation Period', validationPeriod, '< 2',
        #    validationPeriod = 2

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
            print '\nStarting new simulation run from', validationFirstYear, 'to', validationFinalYear
            #init
            model_metrics = init_report()       
            sstDictDF1_ = {} 
            SMdict={}
            #DPScycle+=1
            
            for signal in signal_types:
                if signal[:2] != 'BH' and signal[:2] != 'SH':
                    for m in models[1:]:
                        #for wfStep in wfSteps:
                        wfStep=int(signal.split('_')[0][2:])
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
                                    'ROCLookback': ROCLookback, 'DD95_limit':PRT['DD95_limit'],'tailRiskPct':PRT['tailRiskPct'],\
                                    'initial_equity':PRT['initial_equity'],'horizon':PRT['horizon'],'maxLeverage':PRT['maxLeverage'],\
                                    'CAR25_threshold':PRT['CAR25_threshold'], 'wf_is_period':wf_is_period,'perturbDataPct':perturbDataPct,\
                                    'barSizeSetting':barSizeSetting, 'version':version, 'version_':version_,'maxReadLines':maxReadLines}
                            runName = ticker+'_'+barSizeSetting+'_'+data_type+'_'+filterName+'_'\
                                        + m[0]+'_i'+str(wf_is_period)+'_fcst'+str(wfStep)+'_'+signal
                            model_metrics, sstDictDF1_[runName], SMdict[runName] = wf_classify_validate2(unfilteredData,\
                                                                                        dataSet, m, model_metrics,\
                                                                                        metaData, showPDFCDF=showPDFCDF,\
                                                                                        verbose=verbose)
                else:
                    #buy/sell hold
                    wfStep=int(signal.split('_')[0][2:])
                    wf_is_period = 0
                    m=('None','None')

                        
                    metaData = {'ticker':ticker, 't_start':testFirstYear, 't_end':testFinalYear,\
                            'signal':signal, 'data_type':data_type,'filter':filterName, 'input_signal':input_signal,\
                            'test_split':0, 'iters':1, 'tox_adj':tox_adj_proportion,'longMemory':longMemory,\
                            'n_features':nfeatures, 'FS':feature_selection,'rfe_model':RFE_estimator[0],\
                            'v_start':validationFirstYear, 'v_end':validationFinalYear,'wf_step':wfStep,\
                            'RSILookback': RSILookback,'zScoreLookback': zScoreLookback,'ATRLookback': ATRLookback,\
                            'beLongThreshold': beLongThreshold,'DPOLookback': DPOLookback,'ACLookback': ACLookback,\
                            'CCLookback': CCLookback,'rStochLookback': rStochLookback,'statsLookback': statsLookback,\
                            'ROCLookback': ROCLookback, 'DD95_limit':PRT['DD95_limit'],'tailRiskPct':PRT['tailRiskPct'],\
                            'initial_equity':PRT['initial_equity'],'horizon':PRT['horizon'],'maxLeverage':PRT['maxLeverage'],\
                            'CAR25_threshold':PRT['CAR25_threshold'], 'wf_is_period':wf_is_period,'perturbDataPct':perturbDataPct,\
                            'barSizeSetting':barSizeSetting, 'version':version, 'version_':version_,'maxReadLines':maxReadLines}
                    runName = ticker+'_'+barSizeSetting+'_'+data_type+'_'+filterName+'_'\
                                        + m[0]+'_i'+str(wf_is_period)+'_fcst'+str(wfStep)+'_'+signal
                    model_metrics, sstDictDF1_[runName], SMdict[runName] = wf_classify_validate2(unfilteredData,\
                                                                                dataSet, m, model_metrics,\
                                                                                metaData, showPDFCDF=showPDFCDF,\
                                                                                verbose=verbose)
            #score models
            scored_models, bestModelParams = directional_scoring(model_metrics,filterName)
            #bestModelParams['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            #bestModelParams = bestModelParams.append(pd.Series(data=datetime.datetime.fromtimestamp(time.time())\
            #                               .strftime('%Y-%m-%d %H:%M:%S'), index=['timestamp']))
                                            
            #keep original for other DF
            #sstDictDF1_Combined_DF1_beShorts_ = copy.deepcopy(sstDictDF1_)
            #sstDictDF1_DF1_Shorts_beFlat_ = copy.deepcopy(sstDictDF1_)
            if showAllCharts:
                for runName in sstDictDF1_:
                    compareEquity(sstDictDF1_[runName],runName)
                    
            if bestModelParams.signal[:2] != 'BH' and bestModelParams.signal[:2] != 'SH':
                print  '\nBest signal found...', bestModelParams.signal
                for m in models:
                    if bestModelParams['params'] == str(m[1]):
                        print  'Best model found...\n', m[1]
                        bm = m[1]
                print 'Feature selection: ', bestModelParams.FS
                print 'Number of features: ', bestModelParams.cols
                print 'WF In-Sample Period:', bestModelParams.wf_is_period
                print 'WF Out-of-Sample Period:', bestModelParams.wf_step
                print 'Long Memory: ', longMemory         
                DF1_BMrunName = ticker+'_'+barSizeSetting+'_'+bestModelParams.data_type+'_'+filterName+'_'  +\
                                    bestModelParams.model + '_i'+str(bestModelParams.wf_is_period)\
                                    +'_fcst'+str(bestModelParams.wf_step)+'_'+bestModelParams.signal
            else:
                print  '\nBest signal found...', bestModelParams.signal       
                print  'Model: ', bestModelParams.model       
                DF1_BMrunName = ticker+'_'+barSizeSetting+'_'+bestModelParams.data_type+'_'+filterName+'_'  +\
                                    bestModelParams.model + '_i'+str(bestModelParams.wf_is_period)\
                                    +'_fcst'+str(bestModelParams.wf_step)+'_'+bestModelParams.signal
                                    
            if showBestCharts:
                compareEquity(sstDictDF1_[DF1_BMrunName],'Best Optimization  '+DF1_BMrunName)
            elif runDPS == False:
                #charts off and dps off -> live with no dps
                compareEquity(sstDictDF1_[DF1_BMrunName],DF1_BMrunName, savePath=chartSavePath,\
                                showChart=False, ticker=ticker)    
                                
            #use best params for next step
            for i in range(0,bestModelParams.shape[0]):
                metaData[bestModelParams.index[i]]=bestModelParams[bestModelParams.index[i]]
            
            print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
            
            #DPS
            if runDPS:               
                print '\nDynamic Position Sizing..'
                DPS_both = {}
                for wl in windowLengths:
                    if validationPeriod in validationDict_DPS and \
                           len([x for x in validationDict_DPS[validationPeriod].index\
                                if x not in sstDictDF1_[DF1_BMrunName].index]) > wl:
                        #prior validation data available from the third DPScycle. wl<1st cycle vp
                        sst_bestModel = validationDict_DPS[validationPeriod].drop(['safef','CAR25','dd95','ddTol','system'], axis=1)
                        zero_index = np.array([x for x in sst_bestModel.index if x not in sstDictDF1_[DF1_BMrunName].index])[-int(wl):]
                        sst_zero = sst_bestModel.ix[zero_index]
                        sst_bestModel = pd.concat([sst_zero,sstDictDF1_[DF1_BMrunName]],axis=0)
                        
                        #adjust window length if it is larger than zero index
                        if sst_zero.shape[0]<wl:
                            wl=sst_zero.shape[0]
                    else:
                        #add zeros if first cycle
                        zero_index = dataSet.ix[:sstDictDF1_[DF1_BMrunName].index[0]].index[:-1]
                        sst_zero = pd.concat([pd.Series(data=0, name='signals', index=zero_index ), dataSet.gainAhead.ix[zero_index],\
                                                        dataSet.prior_index.ix[zero_index]], axis=1)
                        sst_bestModel = pd.concat([sst_zero,sstDictDF1_[DF1_BMrunName]],axis=0)
                        
                        #adjust window length if it is larger than zero index
                        if sst_zero.shape[0]<wl:
                            wl=sst_zero.shape[0]

                    #calc DPS
                    startDate = sstDictDF1_[DF1_BMrunName].index[0]
                    endDate = sst_bestModel.index[-1]
                    for ml in maxLeverage:
                        PRT['maxLeverage'] = ml               
                        dpsRun, sst_save = calcDPS2(DF1_BMrunName, sst_bestModel, PRT, startDate,\
                                                                    endDate, wl, 'both', threshold=PRT['CAR25_threshold'])
                        DPS_both[dpsRun] = sst_save
                        
                        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                        #DPS[dpsRun] = sst_save
                v3tag='Best Optimization '    
                dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                                            endDate,'both', DF1_BMrunName, yscale='linear',\
                                                            ticker=ticker,displayCharts=showBestCharts,\
                                                            equityStatsSavePath=equityStatsSavePath, verbose=verbose,
                                                            v3tag=v3tag, returnNoDPS=returnNoDPS)
                bestBothDPS.index.name = 'dates'
                bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system')], \
                                                                axis=1)
                #save best windowlength to metadata
                if dpsRunName.find('wl') != -1:
                    metaData['windowLength']=float(dpsRunName[dpsRunName.find('wl'):].split()[0][2:])
                    metaData['maxLeverage']=int(dpsRunName[dpsRunName.find('maxL'):].split()[0][4:])
                else:
                    #no dps
                    metaData['windowLength']=0
                    metaData['maxLeverage']=1
                    
                if endOfData == 1:
                    #dps and end of data
                    if returnNoDPS == False:
                        #get v3 signal best of either DPS or no DPS, if returnNoDPS is False
                        v3tag=version+' '+ticker+' Signal OOS'    
                        dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                                                    endDate,'both', DF1_BMrunName, yscale='linear',\
                                                                    ticker=ticker,displayCharts=showBestCharts,\
                                                                    equityStatsSavePath=equityStatsSavePath, verbose=verbose,\
                                                                    v3tag=v3tag, returnNoDPS=True, savePath=chartSavePath,\
                                                                    numCharts=1)
                        bestBothDPS.index.name = 'dates'
                        bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system')], \
                                                                        axis=1)                       
                        #save sst's for save file         
                        v3signals[validationPeriod] = bestBothDPS
                    else:
                        #NoDPS is being returned
                        v3signals[validationPeriod] = bestBothDPS
                        
                    #save model
                    BSMdict[validationPeriod] = SMdict[DF1_BMrunName]
                    
                #if showAllCharts:
                #    DPSequity = calcEquity_df(bestBothDPS[['signals','gainAhead']], v3tag+dpsRunName, leverage = bestBothDPS.safef.values)
            
            elif endOfData == 1:
                #nodps and end of data
                v3signals[validationPeriod] = pd.concat([sstDictDF1_[DF1_BMrunName],\
                                    pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=DF1_BMrunName, name = 'system', index = sstDictDF1_[DF1_BMrunName].index)
                                    ],axis=1)
                #save model
                BSMdict[validationPeriod] = SMdict[DF1_BMrunName]
            else:
                #no dps and not end of data
                pass

             
            #adjust test dates
            t_start_loc = dataSet.reset_index()[dataSet.reset_index().dates ==bestModelParams.v_end].index[0]
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
                print '\nSimulating Next Period from', validationFirstYear, 'to',validationFinalYear
                #update metaData dates
                metaData['t_start']=testFirstYear
                metaData['t_end']=testFinalYear
                metaData['v_start']=validationFirstYear
                metaData['v_end']=validationFinalYear
                nextRunName = 'Next'+str(validationPeriod)+'_'+DF1_BMrunName
                m = (bestModelParams.model,eval(bestModelParams.params))
                print 'using', m
                model_metrics, sstDictDF1_[nextRunName], savedModel = wf_classify_validate2(unfilteredData, dataSet, m, model_metrics,\
                                                    metaData, showPDFCDF=showPDFCDF, verbose=verbose)
                if showBestCharts:
                    compareEquity(sstDictDF1_[nextRunName],nextRunName)
                    
                #save noDPS validation Curve
                if validationPeriod not in validationDict_noDPS:
                    validationDict_noDPS[validationPeriod] =pd.concat([sstDictDF1_[nextRunName],\
                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                            ],axis=1)
                                            
                    if sum(np.isnan(validationDict_noDPS[validationPeriod].prior_index.values))>0:
                        message="Error: NaN in bestBothDPS.prior_index:3 "
                        offlineMode(ticker, message, signalPath, version, version_)   
                        
                else:
                    validationDict_noDPS[validationPeriod] =validationDict_noDPS[validationPeriod].append(                                                                         
                                              pd.concat([sstDictDF1_[nextRunName],\
                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                            ],axis=1)                                                               
                                             )
                                             
                    if sum(np.isnan(validationDict_noDPS[validationPeriod].prior_index.values))>0:
                        message="Error: NaN in bestBothDPS.prior_index:4"
                        offlineMode(ticker, message, signalPath, version, version_)  

                print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
                #print 'Finished Next Period Model Training... '   
                
                if runDPS:
                    #if DPS was chosen as the best model to use going forward
                    DPS_both = {}
                    if dpsRunName.find('wl') != -1:
                        #for ml in maxLeverage:
                        PRT['maxLeverage'] = float(dpsRunName[dpsRunName.find('wl'):].split()[1][4:])
                        #for wl in windowLengths:
                        wl=float(dpsRunName[dpsRunName.find('wl'):].split()[0][2:])
                        
                        #print 'Beginning Next Period Dynamic Position Sizing..'
                        if validationPeriod in validationDict_DPS and validationDict_DPS[validationPeriod].shape[0]>=wl:
                            sst_bestModel = pd.concat([validationDict_DPS[validationPeriod].drop(['safef','CAR25','dd95','ddTol','system'],axis=1),\
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
                        startDate = sstDictDF1_[nextRunName].index[0]
                        endDate = sst_bestModel.index[-1]
                        dpsRun, sst_save = calcDPS2(nextRunName, sst_bestModel, PRT, startDate,\
                                                                    endDate, wl, 'both', threshold=PRT['CAR25_threshold'])
                        DPS_both[dpsRun] = sst_save
                        
                        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                        #DPS[dpsRun] = sst_save
                        v3tag='Next Period '
                        dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                                                    endDate,'both', nextRunName, yscale='linear',\
                                                                    ticker=ticker,displayCharts=showBestCharts,\
                                                                    equityStatsSavePath=equityStatsSavePath, verbose=verbose,
                                                                    v3tag=v3tag, returnNoDPS=returnNoDPS)
                        bestBothDPS.index.name = 'dates'
                        bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system'),\
                                                                ], axis=1)
                        #if showAllCharts:                                       
                        #    DPSequity = calcEquity_df(bestBothDPS[['signals','gainAhead']], v3tag+dpsRunName, leverage = bestBothDPS.safef.values)
                        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
                        #print 'Finished Dynamic Position Sizing..'
                        
                        if validationPeriod not in validationDict_DPS:
                            validationDict_DPS[validationPeriod] =bestBothDPS
                            if sum(np.isnan(validationDict_DPS[validationPeriod].prior_index.values))>0:
                                    message = "Error: NaN in bestBothDPS.prior_index:1 "
                                    offlineMode(ticker, message, signalPath, version, version_)
                        else:              
                            validationDict_DPS[validationPeriod] =validationDict_DPS[validationPeriod].append(bestBothDPS)
                            if sum(np.isnan(validationDict_DPS[validationPeriod].prior_index.values))>0:
                                    message="Error: NaN in bestBothDPS.prior_index:2 "
                                    offlineMode(ticker, message, signalPath, version, version_)                               

          
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print '\n\nScoring Validation Curves...'
    #set validation curves equal length. end date for CAR25/metrics is -2 because GA at [-1] is 0.
    vStartDate = dataSet.index[0]
    vEndDate = dataSet.index[-2]
    for validationPeriod in validationDict_noDPS:
        if validationDict_noDPS[validationPeriod].index[0]>vStartDate:
            vStartDate =  validationDict_noDPS[validationPeriod].index[0]
            
    vCurve_metrics_noDPS = init_report()
    for validationPeriod in validationDict_noDPS:
        vCurve = 'No DPS '+ticker+'_'+barSizeSetting+' Validation Period '+str(validationPeriod)
        if showBestCharts:
            compareEquity(validationDict_noDPS[validationPeriod],vCurve)
        if scorePath is not None:
            validationDict_noDPS[validationPeriod].to_csv(equityStatsSavePath+vCurve+'_'+\
                                                            str(validationDict_noDPS[validationPeriod].index[0]).replace(':','')+'_to_'+\
                                                            str(validationDict_noDPS[validationPeriod].index[-1]).replace(':','')+'.csv')
        CAR25_oos = CAR25_df_bars(vCurve,validationDict_noDPS[validationPeriod].ix[vStartDate:vEndDate].signals,\
                                            validationDict_noDPS[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int),\
                                            unfilteredData.Close, DD95_limit =PRT['DD95_limit'],barSize=barSizeSetting,\
                                            number_forecasts=50, fraction=max(maxLeverage))
        model = [validationDict_noDPS[validationPeriod].iloc[-1].system.split('_')[5],\
                        [m[1] for m in models if m[0] ==validationDict_noDPS[validationPeriod].iloc[-1].system.split('_')[5]][0]]
        #metaData['model'] = model[0]
        #metaData['params'] =model[1]
        metaData['validationPeriod'] = validationPeriod
        vCurve_metrics_noDPS = update_report(vCurve_metrics_noDPS, filterName,\
                                            validationDict_noDPS[validationPeriod].ix[vStartDate:vEndDate].signals.values.astype(int), \
                                            dataSet.ix[validationDict_noDPS[validationPeriod].ix[vStartDate:vEndDate].index].signal.values.astype(int),\
                                            unfilteredData.gainAhead,\
                                            validationDict_noDPS[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int), model,\
                                            metaData,CAR25_oos)
    #score models
    scored_models_noDPS, bestModelParams_noDPS = directional_scoring(vCurve_metrics_noDPS,filterName)
    
    #bestModelParams['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    if scorePath is not None:
        scored_models_noDPS.to_csv(scorePath+version_+'_'+ticker+'_'+barSizeSetting+'_noDPS.csv')
        
    print '\nBest no DPS Validation Period Found:', bestModelParams_noDPS.validationPeriod    
    if showBestCharts:
        BestEquity = calcEquity_df(validationDict_noDPS[bestModelParams_noDPS.validationPeriod][['signals','gainAhead']],\
                                            'Best No DPS '+bestModelParams_noDPS.C25sig, leverage = validationDict_noDPS[bestModelParams_noDPS.validationPeriod].safef.values)
    saveParams(dataSet, bestModelParams_noDPS, bestParamsPath, modelSavePath, 'v1',barSizeSetting)
    
    if runDPS:
        vCurve_metrics_DPS = init_report()
        for validationPeriod in validationDict_DPS:
            vCurve = 'DPS '+ticker+'_'+barSizeSetting+' Validation Period '+str(validationPeriod)
            #compareEquity(validationDict_DPS[validationPeriod],vCurve)
            if showBestCharts:
                DPSequity = calcEquity_df(validationDict_DPS[validationPeriod][['signals','gainAhead']], vCurve,\
                                                    leverage = validationDict_DPS[validationPeriod].safef.values)
            if scorePath is not None:
                validationDict_DPS[validationPeriod].to_csv(equityStatsSavePath+vCurve+'_'+\
                                                                str(validationDict_DPS[validationPeriod].index[0]).replace(':','')+'_to_'+\
                                                                str(validationDict_DPS[validationPeriod].index[-1]).replace(':','')+'.csv')
            CAR25_oos = CAR25_df_bars(vCurve,validationDict_DPS[validationPeriod].ix[vStartDate:vEndDate].signals,\
                                                validationDict_DPS[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int),\
                                                unfilteredData.Close, DD95_limit =PRT['DD95_limit'],barSize=barSizeSetting,\
                                                number_forecasts=50, fraction=max(maxLeverage))
            model = [validationDict_DPS[validationPeriod].iloc[-1].system.split('_')[5],\
                            [m[1] for m in models if m[0] ==validationDict_DPS[validationPeriod].iloc[-1].system.split('_')[5]][0]]
            metaData['validationPeriod'] = validationPeriod
            #metaData['params'] =model[1]
            vCurve_metrics_DPS = update_report(vCurve_metrics_DPS, filterName,\
                                                validationDict_DPS[validationPeriod].ix[vStartDate:vEndDate].signals.values.astype(int), \
                                                dataSet.ix[validationDict_DPS[validationPeriod].ix[vStartDate:vEndDate].index].signal.values.astype(int),\
                                                unfilteredData.gainAhead,\
                                                validationDict_DPS[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int), model,\
                                                metaData,CAR25_oos)        
        #score models
        scored_models_DPS, bestModelParams_DPS = directional_scoring(vCurve_metrics_DPS,filterName)
        
        #bestModelParams['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        if scorePath is not None:
            scored_models_DPS.to_csv(scorePath+version_+'_'+ticker+'_'+barSizeSetting+'_DPS.csv')
            
        print '\nBest DPS Validation Period Found:', bestModelParams_DPS.validationPeriod        
        if showBestCharts:
            BestEquity = calcEquity_df(validationDict_DPS[bestModelParams_DPS.validationPeriod][['signals','gainAhead']],\
                                                'Best DPS '+bestModelParams_DPS.C25sig, leverage = validationDict_DPS[bestModelParams_DPS.validationPeriod].safef.values)
        saveParams(dataSet, bestModelParams_DPS, bestParamsPath, modelSavePath, 'v2', barSizeSetting)


    print version_, 'Saving Signals..'      
    timenow, lastBartime, cycleTime = getCycleTime(dataSet)
    #v3signals[bestModelParams_noDPS.validationPeriod].tail().to_csv(signalPath + version+'_'+ ticker+'_'+barSizeSetting+ '.csv')
    files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
    
    #old version file
    if version+'_'+ ticker+ '.csv' not in files:
        signalFile = v3signals[bestModelParams_noDPS.validationPeriod].tail()
        signalFile.to_csv(signalPath + version+'_'+ ticker+ '.csv', index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
        addLine = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1].name
        signalFile = signalFile.append(addLine)
        signalFile.to_csv(signalPath + version+'_'+ ticker+ '.csv', index=True)
        
    #new version file
    if version_+'_'+ ticker+'_'+barSizeSetting+ '.csv' not in files:
        #signalFile = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-2:]
        addLine = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1].name
        signalFile = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-2:-1].append(addLine)
        signalFile.index.name = 'dates'
        signalFile.to_csv(signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index_col=['dates'])
        addLine = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1].name
        signalFile = signalFile.append(addLine)
        signalFile.to_csv(signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index=True)
    
    print '\n'+version, 'Next Signal:'
    print v3signals[bestModelParams_noDPS.validationPeriod].iloc[-1].system
    print v3signals[bestModelParams_noDPS.validationPeriod].drop(['system'],axis=1).iloc[-1]
    
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Validation Period Parameter Search!'
    #message="v3 signals offline"
    #offlineMode(ticker, message, signalPath, version, version_) 
    ##############################################################
