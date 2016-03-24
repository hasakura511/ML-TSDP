
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015
changelog
v2.31C " MJ"
added compatibility with v3
fixed timestamp

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
                         directional_scoring, compareEquity, describeDistribution
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df,\
                            maxCAR25, wf_classify_validate, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
from suztoolz.transform import zigzag as zg
from suztoolz.data import getDataFromIB

start_time = time.time()
#system parameters
version = 'v2'
version_ = 'v2.32'
filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='1 min'

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
                    #'AUDUSD',\
                    #'EURUSD',\
                    #'GBPUSD',\
                    #'USDCAD',\
                    #'USDCHF',\
                    #'NZDUSD',
                    #'EURCHF',\
                    'EURGBP'\
                    ]
                    
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    runDPS = False
    #scorePath = './debug/scored_metrics_'
    #equityStatsSavePath = './debug/'
    #signalPath = './debug/'
    #dataPath = './data/from_IB/'
    scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/'
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    
else:
    print 'Live Mode', sys.argv[1], sys.argv[2]
    debug=False
    
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    runDPS = False
    
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    bestParamsPath =  './data/params/'
    
    if sys.argv[2] == "0":
        livePairs=[]
        pair = sys.argv[1]
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if version+'_'+ pair + '.csv' in files:
            signalFile=pd.read_csv(signalPath+ version+'_'+ pair + '.csv', parse_dates=['dates'])
            offline = signalFile.iloc[-1].copy(deep=True)
            offline.dates = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.signals = 0
            offline.gainAhead =0
            offline.prior_index=0
            offline.safef=0
            offline.CAR25=0
            offline.dd95 = 0
            offline.ddTol=0
            offline.system = 'Offline'
            signalFile=signalFile.append(offline)
            signalFile.to_csv(signalPath + version+'_'+ pair + '.csv', index=False)
            
        sys.exit("Offline Mode: "+sys.argv[0]+' '+sys.argv[1])
    else:
        livePairs=[sys.argv[1]]



for ticker in livePairs:
    print 'Begin optimization run for', ticker
    symbol=ticker[0:3]
    currency=ticker[3:6]
    
    #load best params
    bestParams = pd.read_csv(bestParamsPath+'v3_'+ticker+'.csv')
    validationStartPoint = bestParams.iloc[-1].validationPeriod
    #validationStartPoint = 250
    
    #Model Parameters
    #signal_types = ['gainAhead','ZZ']
    #signal_types = ['ZZ']
    signal_types = ['gainAhead']    
    zz_steps = [0.001,0.002,0.003]
    perturbDataPct = 0.0002
    longMemory =  False
    iterations=1
    input_signal = 1
    feature_selection = 'None' #RFECV OR Univariate
    #feature_selection = 'Univariate' #RFECV OR Univariate
    wfSteps=[5,10]
    wf_is_periods = [250,500,1000]
    #wf_is_periods = [100]
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
    windowLengths = [30]
    maxLeverage = [5]
    PRT={}
    PRT['DD95_limit'] = 0.05
    PRT['tailRiskPct'] = 95
    PRT['initial_equity'] = 1.0
    PRT['horizon'] = 250
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
             #("LR", LogisticRegression(class_weight={1:500})), \
             #("PRCEPT", Perceptron(class_weight={1:500})), \
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
    
    if 'ZZ' in signal_types:
        signal_types.remove('ZZ')
        print 'Creating Signal labels..',
        for i in zz_steps:
            #for j in zz_steps:
            label = 'ZZ '+str(i) + ',-' + str(i)
            print label,
            signal_types.append(label)
            #zz_signals[label] = zg(dataSet.Close, i, -j).pivots_to_modes()

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


    print '\n\nNew simulation run... '
    if showDist:
        describeDistribution(dataSet.reset_index()['Close'], dataSet.reset_index()['priceChange'], ticker)

    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Pre-processing... Beginning Model Training..'

    #########################################################
    model_metrics = init_report()       
    sstDictDF1_ = {}
    
    for signal in signal_types:
        for i,m in enumerate(models):
            for wfStep in wfSteps:
                for wf_is_period in wf_is_periods:
                    testFirstYear = dataSet.index[0]
                    testFinalYear = dataSet.index[max(wf_is_periods)-1]
                    validationFirstYear =dataSet.index[max(wf_is_periods)]
                    validationFinalYear =dataSet.index[-1]
                    
                    if validationStartPoint is not None and\
                                dataSet.shape[0]>dataSet.index[-validationStartPoint:].shape[0]+max(wf_is_periods):
                        testFinalYear = dataSet.index[-validationStartPoint-1]
                        validationFirstYear =dataSet.index[-validationStartPoint]

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
                    model_metrics, sstDictDF1_[runName] = wf_classify_validate(dataSet, dataSet, [m], model_metrics,\
                                                        wf_is_period, metaData, PRT, showPDFCDF=showPDFCDF,longMemory=longMemory)

    #score models
    scored_models, bestModel = directional_scoring(model_metrics,filterName)
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
        scored_models.to_csv(scorePath+filterName+'.csv')

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
        
    bestSSTnoDPS = pd.concat([sstDictDF1_[DF1_BMrunName],\
                    pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[DF1_BMrunName].index),
                    pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[DF1_BMrunName].index),
                    pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[DF1_BMrunName].index),
                    pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[DF1_BMrunName].index),
                    pd.Series(data=DF1_BMrunName, name = 'system', index = sstDictDF1_[DF1_BMrunName].index)
                    ],axis=1)    
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Model Training... Beginning Dynamic Position Sizing..'
    ##############################################################

    if runDPS:
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
                    
                dpsRun, sst_save = calcDPS2(DF1_BMrunName, sst_bestModel, PRT, startDate, endDate, wl, 'both', threshold=CAR25_threshold)
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
        print '\n'+version, 'Next Signal:'
        print bestBothDPS.iloc[-1].system
        print bestBothDPS.drop(['system'],axis=1).iloc[-1]
         
        print 'Saving Signals..'      
        #init file
        #bestBothDPS.tail().to_csv(signalPath + version+'_'+ ticker + '.csv')
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if version+'_'+ ticker + '.csv' not in files:
            signalFile = bestBothDPS.tail()
            signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version+'_'+ ticker + '.csv', parse_dates=['dates'])

            #if bestBothDPS.reset_index().iloc[-1].dates != signalFile.iloc[-1].dates:
            if bestBothDPS.reset_index().iloc[-2].dates == signalFile.iloc[-1].dates:
                signalFile.gainAhead.iloc[-1] == bestBothDPS.gainAhead.iloc[-2]
            signalFile=signalFile.append(bestBothDPS.reset_index().iloc[-1])
            signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=False)
            
        #new version file
        if version_+'_'+ ticker + '.csv' not in files:
            #signalFile = bestBothDPS.iloc[-2:]
            addLine = bestBothDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestBothDPS.iloc[-1].name
            signalFile = bestBothDPS.iloc[-2:-1].append(addLine)
            signalFile.index.name = 'dates'
            signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker + '.csv', index_col=['dates'])
            addLine = bestBothDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestBothDPS.iloc[-1].name
            signalFile = signalFile.append(addLine)
            signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
    else:
        print 'Saving Signals..'      
        #init file
        #bestSSTnoDPS.tail().to_csv(signalPath + version+'_'+ ticker + '.csv')
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if version+'_'+ ticker + '.csv' not in files:
            signalFile = bestSSTnoDPS.tail()
            signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version+'_'+ ticker + '.csv', parse_dates=['dates'])

            #if bestSSTnoDPS.reset_index().iloc[-1].dates != signalFile.iloc[-1].dates:
            if bestSSTnoDPS.reset_index().iloc[-2].dates == signalFile.iloc[-1].dates:
                signalFile.gainAhead.iloc[-1] == bestSSTnoDPS.gainAhead.iloc[-2]
            signalFile=signalFile.append(bestSSTnoDPS.reset_index().iloc[-1])
            signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=False)
            
        #new version file
        if version_+'_'+ ticker + '.csv' not in files:
            #signalFile = bestSSTnoDPS.iloc[-2:]
            addLine = bestSSTnoDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestSSTnoDPS.iloc[-1].name
            signalFile = bestSSTnoDPS.iloc[-2:-1].append(addLine)
            signalFile.index.name = 'dates'
            signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
        else:        
            signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker + '.csv', index_col=['dates'])
            addLine = bestSSTnoDPS.iloc[-1]
            addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            addLine.name = bestSSTnoDPS.iloc[-1].name
            signalFile = signalFile.append(addLine)
            signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
            
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
