
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015
changelog
v2.4C " MJ"
added compatibility with v3
fixed timestamp
added wf stepping for ga and zz signals

v2.3C " MJ"
fixed offline mode
added params folder
added sysargv
added signal types
fixed future leak in zz signals
added ValidationDataPoints
fixed signal file timestamp duplication
fixed gainAhead appending to previous timestamp
creates new signal file if one dosent exist for current version
added online/offline mode
one script to cycle through all currency pairs

v2.20C
moved getData from IB to another script
added zigzag signals
added walkforward classifier

v2.10
added other pairs as features
added Baysian Ridge as model option

v2.01
added roofing filter indicators

v2.0
added in-sample period walk forward optemization
added dps optimization
added perturb data for robustness


@author: hidemi
"""
import math
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
import arch

#suztoolz
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve,\
                         directional_scoring, compareEquity, describeDistribution,\
                         offlineMode
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df,\
                            maxCAR25, wf_classify_validate2, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter, getCycleTime
                        
from suztoolz.transform import zigzag as zg
from suztoolz.data import getDataFromIB

start_time = time.time()
#system parameters
version = 'v2'
version_ = 'v2.4C'

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='30m'

currencyPairs = ['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']
                
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
                    'AUDUSD',\
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
    showAllCharts = False
    perturbData = False
    runDPS = True
    saveParams = False
    saveDataSet = True
    verbose = False
    #scorePath = './debug/scored_metrics_'
    #equityStatsSavePath = './debug/'
    #signalPath = './debug/'
    #dataPath = './data/from_IB/'
    scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/' 
    
else:
    print 'Live Mode', sys.argv[1], sys.argv[2]
    debug=False
    
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    runDPS = True
    saveParams = False
    scorePath = None
    equityStatsSavePath = None
    saveDataSet=True
    verbose= False
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    bestParamsPath =  './data/params/'
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
    
    #load best params
    bestParams = pd.read_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv').iloc[-1]
    maxReadLines=int(bestParams.maxReadLines)
    #Model Parameters
    validationStartPoint = bestParams.validationPeriod
    signal_types = [bestParams.signal]
 
    wf_is_periods = [bestParams.wf_is_period]
    perturbDataPct = bestParams.perturbDataPct
    longMemory =  bestParams.longMemory
    iterations=1
    input_signal = bestParams.input_signal
    feature_selection = bestParams.FS
    tox_adj_proportion = bestParams.tox_adj
    nfeatures = bestParams.n_features

    #feature Parameters
    RSILookback = bestParams.RSILookback
    zScoreLookback = bestParams.zScoreLookback
    ATRLookback = bestParams.ATRLookback
    beLongThreshold = bestParams.beLongThreshold
    DPOLookback = bestParams.DPOLookback
    ACLookback = bestParams.ACLookback
    CCLookback = bestParams.CCLookback
    rStochLookback = bestParams.rStochLookback
    statsLookback = bestParams.statsLookback
    ROCLookback = bestParams.ROCLookback
    models = [(bestParams.model,eval(bestParams.params))]

    #DPS parameters
    if bestParams.windowLength == 0:
        windowLengths = []
    else:
        windowLengths = [bestParams.windowLength]    
    maxLeverage = [bestParams.maxLeverage]
    #personal risk tolerance parameters
    PRT={}
    PRT['DD95_limit'] =bestParams.DD95_limit
    PRT['tailRiskPct'] =bestParams.tailRiskPct
    PRT['initial_equity'] =bestParams.initial_equity
    PRT['horizon'] =bestParams.horizon
    PRT['maxLeverage'] =bestParams.maxLeverage
    PRT['CAR25_threshold'] =bestParams.CAR25_threshold
    #CAR25_threshold=-np.inf
    #CAR25_threshold=0

    #model selection
    
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    RFE_estimator = [ 
            ("None","None"),\
            #("GradientBoostingRegressor",GradientBoostingRegressor()),\
            #("DecisionTreeRegressor",DecisionTreeRegressor()),\
            #("ExtraTreeRegressor",ExtraTreeRegressor()),\
            #("BayesianRidge", BayesianRidge()),\
             ]
    '''         
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
             #("LR", LogisticRegression()), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:500})), \
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
             #("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:500}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             ("VotingHard", VotingClassifier(estimators=[\
             #    ("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #    ("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #    ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #    ("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 ("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #    ("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
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
    '''

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
    
    #short direction
    dataSet['Pri_RSI'] = RSI(dataSet.Close,RSILookback)
    dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
    dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
    dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
    dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)

    #long direction
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
    
    '''
    for wfStep in wfSteps:
        print '\nCreating Signal labels..'
        if 'zigZag' in signal_types:
            signal_types.remove('zigZag')
            for i in zz_steps:
                #for j in zz_steps:
                label = 'ZZ'+str(wfStep)+'_'+str(i) + ',-' + str(i)
                print label+',',
                signal_types.append(label)
                #zz_signals[label] = zg(dataSet.Close, i, -j).pivots_to_modes()
                
        if 'gainAhead' in signal_types:
            signal_types.remove('gainAhead')
            label = 'GA'+str(wfStep)
            print label+',',
            signal_types.append(label)
        
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
    '''
    
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


    print '\n\nNew simulation run... '
    if showDist:
        describeDistribution(dataSet.reset_index()['Close'], dataSet.reset_index()['priceChange'], ticker)

    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Pre-processing... Beginning Model Training..'

    #########################################################
    model_metrics = init_report()       
    sstDictDF1_ = {}
    SMdict={}
    
    testFirstYear = dataSet.index[0]
    testFinalYear = dataSet.index[max(wf_is_periods)-1]
    validationFirstYear =dataSet.index[max(wf_is_periods)]
    validationFinalYear =dataSet.index[-1]
    
    if validationStartPoint is not None and\
                dataSet.shape[0]>dataSet.index[-validationStartPoint:].shape[0]+max(wf_is_periods)+max(windowLengths):
        testFinalYear = dataSet.index[-validationStartPoint-max(windowLengths)-1]
        validationFirstYear =dataSet.index[-validationStartPoint-max(windowLengths)]    
    
    for signal in signal_types:
        if signal[:2] != 'BH' and signal[:2] != 'SH':
            wfStep=int(signal.split('_')[0][2:])
            for i,m in enumerate(models):          
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
                    runName = ticker+'_'+barSizeSetting+'_'+data_type+'_'+filterName+'_' + m[0]+\
                                            '_i'+str(wf_is_period)+'_fcst'+str(wfStep)+'_'+signal
                    model_metrics, sstDictDF1_[runName], SMdict[runName] = wf_classify_validate2(unfilteredData,\
                                                                        dataSet, m, model_metrics,\
                                                                        metaData, showPDFCDF=showPDFCDF,verbose=verbose,\
                                                                        PDFCDFsavePath=chartSavePath,\
                                                                        PDFCDFfilename=version+' '+ticker)
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
                                                                        verbose=verbose,\
                                                                        PDFCDFsavePath=chartSavePath,\
                                                                        PDFCDFfilename=version+' '+ticker)  
    #score models
    scored_models, bestModelParams = directional_scoring(model_metrics,filterName)
    
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
    if showAllCharts:
        compareEquity(sstDictDF1_[DF1_BMrunName],DF1_BMrunName)       
    elif runDPS == False:
        compareEquity(sstDictDF1_[DF1_BMrunName],DF1_BMrunName, savePath=chartSavePath,\
                                showChart=False, ticker=ticker)
        
    bestSSTnoDPS = pd.concat([sstDictDF1_[DF1_BMrunName],\
                pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[DF1_BMrunName].index),
                pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[DF1_BMrunName].index),
                pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[DF1_BMrunName].index),
                pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[DF1_BMrunName].index),
                pd.Series(data=DF1_BMrunName, name = 'system', index = sstDictDF1_[DF1_BMrunName].index)
                ],axis=1)      
        
    if scorePath is not None:
        scored_models.to_csv(scorePath+filterName+'.csv')
        
    timenow, lastBartime, cycleTime = getCycleTime(dataSet)
    
    if saveParams:
        print version, 'Saving Params..' 
        bestModelParams = bestModelParams.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        bestModelParams = bestModelParams.append(pd.Series(data=lastBartime.strftime("%Y%m%d %H:%M:%S %Z"), index=['lastBarTime']))
        bestModelParams = bestModelParams.append(pd.Series(data=cycleTime, index=['cycleTime']))
                               
        files = [ f for f in listdir(bestParamsPath) if isfile(join(bestParamsPath,f)) ]
        if version+'_'+ticker+'_'+barSizeSetting+ '.csv' not in files:
            BMdf = pd.concat([bestModelParams,bestModelParams],axis=1).transpose()
            #BMdf.index = BMdf.timestamp
            BMdf.to_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv', index=False)
        else:
            BMdf = pd.read_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv').append(bestModelParams, ignore_index=True)
            #BMdf.index = BMdf.timestamp
            BMdf.to_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv', index=False)


    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'

    ##############################################################

    if runDPS:
        print 'Finished Model Training... Beginning Dynamic Position Sizing..'
        sst_bestModel = sstDictDF1_[DF1_BMrunName].copy(deep=True)

        #calc DPS
        DPS = {}
        DPS_both = {}

        startDate = sst_bestModel.index[max(windowLengths)]
        endDate = sst_bestModel.index[-1]
        for ml in maxLeverage:
            PRT['maxLeverage'] = ml
            for wl in windowLengths:
                #sst_bestModel = pd.concat([pd.Series(data=systems[s].signals, name='signals', index=systems[s].index),\
                #                   systems[s].gainAhead], axis=1)

                #newStart = start            
                #if sst_bestModel.ix[:startDate].shape[0] < wl:
                    #zerobegin = pd.DataFrame(data=0, columns= ['signals','gainAhead'],\
                    #    index = [sst_bestModel.index[0] - datetime.timedelta(days=x) for x in range(int(math.ceil(wl)),0,-1)])
                    #newStart = datetime.datetime.strftime(sst_bestModel.iloc[wl].name,'%Y-%m-%d')
                    #print 'both DPS', start,'<=',startDate.date(),'adjusting', start,'to', newStart
                    #sst_bestModel = pd.concat([zerobegin, sst_bestModel], axis=0)
                    
                dpsRun, sst_save = calcDPS2(DF1_BMrunName, sst_bestModel, PRT, startDate, endDate, wl, 'both', threshold=PRT['CAR25_threshold'])
                DPS_both[dpsRun] = sst_save
                
                #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                #DPS[dpsRun] = sst_save
        v3tag=version+' '+ticker+' Signal OOS'
        dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                                    endDate,'both', DF1_BMrunName, yscale='linear',\
                                                    ticker=ticker,displayCharts=showAllCharts,\
                                                    equityStatsSavePath=equityStatsSavePath, verbose=verbose,
                                                    v3tag=v3tag, returnNoDPS=False,savePath=chartSavePath)
        bestBothDPS.index.name = 'dates'
        bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system')], \
                                                        axis=1)
        print '\n'+version, 'Next Signal:'
        print bestBothDPS.iloc[-1].system
        print bestBothDPS.drop(['system'],axis=1).iloc[-1]
         
        print 'Saving Signals..'      
        #old file
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if version+'_'+ ticker+ '.csv' not in files:
            signalFile = bestBothDPS.tail()
            signalFile.to_csv(signalPath + version+'_'+ ticker+ '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
            addLine = bestBothDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestBothDPS.iloc[-1].name
            signalFile = signalFile.append(addLine)
            signalFile.to_csv(signalPath + version+'_'+ ticker+ '.csv', index=True)
            
        #new version file
        if version_+'_'+ ticker+'_'+barSizeSetting+ '.csv' not in files:
            #signalFile = bestBothDPS.iloc[-2:]
            addLine = bestBothDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestBothDPS.iloc[-1].name
            signalFile = bestBothDPS.iloc[-2:-1].append(addLine)
            signalFile.index.name = 'dates'
            signalFile.to_csv(signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index_col=['dates'])
            addLine = bestBothDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestBothDPS.iloc[-1].name
            signalFile = signalFile.append(addLine)
            signalFile.to_csv(signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index=True)
    else:
        #old file
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if version+'_'+ ticker+ '.csv' not in files:
            signalFile = bestSSTnoDPS.tail()
            signalFile.to_csv(signalPath + version+'_'+ ticker+ '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
            addLine = bestSSTnoDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestSSTnoDPS.iloc[-1].name
            signalFile = signalFile.append(addLine)
            signalFile.to_csv(signalPath + version+'_'+ ticker+ '.csv', index=True)
            
            
        #new version file
        if version_+'_'+ ticker+'_'+barSizeSetting+ '.csv' not in files:
            #signalFile = bestSSTnoDPS.iloc[-2:]
            addLine = bestSSTnoDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestSSTnoDPS.iloc[-1].name
            signalFile = bestSSTnoDPS.iloc[-2:-1].append(addLine)
            signalFile.index.name = 'dates'
            signalFile.to_csv(signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index_col=['dates'])
            addLine = bestSSTnoDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestSSTnoDPS.iloc[-1].name
            signalFile = signalFile.append(addLine)
            signalFile.to_csv(signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index=True)
            
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
