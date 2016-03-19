import copy
from scipy import stats
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from IPython.display import Image
import pydot
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFE, RFECV
from sknn.mlp import Classifier, Layer
from sknn.backend import lasagne

def getToxCutoff(p, datapoints):
    # TOP 25 volatility days
    if datapoints > p.shape[0]:
        #print 'ToxCutoff:',datapoints,'min datapoints >',p.shape[0],'rows. returning',p.shape[0],'rows.'
        return p.shape[0], 0
    X2 = np.sort(p)
    F2 = np.array(range(p.shape[0]), dtype=float)/p.shape[0]
    days = 0
    cutoff = 100
    while days < datapoints:
        cutoff = cutoff -5        
        days = int(round(len(X2)*(100-cutoff)/100,0))
        threshold = round(stats.scoreatpercentile(X2,cutoff),6)        
        #print days, cutoff, threshold
    return days, threshold
    
def getToxCutoff2(p, cutoff):
    # TOP 25 volatility days
    #if datapoints > p.shape[0]:
    #    #print 'ToxCutoff:',datapoints,'min datapoints >',p.shape[0],'rows. returning',p.shape[0],'rows.'
    #    return p.shape[0], 0
    X2 = np.sort(p)
    F2 = np.array(range(p.shape[0]), dtype=float)/p.shape[0]
    days = int(round(len(X2)*(cutoff/100.0),0))
    threshold = round(stats.scoreatpercentile(X2,100-cutoff),6)        
    #print days, cutoff, threshold        
    return days, threshold

def adjustDataProportion2(mmData, proportion,verbose=False):
    #non-monte carlo methods to make training faster
    if proportion != 0:
        nrows = mmData.shape[0]
        neg_count=len(mmData.loc[mmData['signal']==-1])
        pos_count=len(mmData.loc[mmData['signal']==1])
        mmData_adj = mmData.loc[mmData['signal']==1]
        nrows_adj = float(mmData_adj.shape[0])
        
        if (neg_count/pos_count) < proportion:
            if verbose:
                print '(neg_count/pos_count) < proportion, sampling with replacement..'
            #raise ValueError('(neg_count/pos_count) < proportion, cannot sample w/o replacement')
            nt_count = 0
            x = True
            while x:
                add_index = random.choice(mmData.loc[mmData['signal']==-1].index) #sample with replacement
                if mmData['signal'][add_index] ==-1:
                    mmData_adj = mmData_adj.append(mmData.iloc[add_index])
                    nt_count = float(mmData_adj.loc[mmData_adj['signal']==-1]['signal'].count())
                    #print nt_count, nrows_adj, nt_count/nrows_adj
                    
                if (nt_count/nrows_adj) >= proportion:
                    x = False
        else:
            if verbose:
                print '(neg_count/pos_count) > proportion, appending recent', pos_count, '-1 days..'
            #append recent neg_count -1 days
            lastNegDaysIndex = mmData.loc[mmData['signal']==-1].index[-pos_count:]
            mmData_adj = pd.concat([mmData_adj, mmData.iloc[lastNegDaysIndex]], axis=0).sort_index()
            '''
            nt_count = 0
            x = True
            while x:
                add_index = random.choice(mmData.loc[mmData['signal']==-1].index) #sample without
                if mmData['signal'][add_index] ==-1 and add_index not in mmData_adj.index:
                    mmData_adj = mmData_adj.append(mmData.iloc[add_index])
                    nt_count = float(mmData_adj.loc[mmData_adj['signal']==-1]['signal'].count())
                    #print nt_count, nrows_adj, nt_count/nrows_adj
                    
                if (nt_count/nrows_adj) >= proportion:
                    x = False
            '''
        if verbose:    
            print "Adjusted Training Set to %i rows..(proportion = %f) " % (mmData_adj.shape[0], proportion)   
        return mmData_adj
    else:
        return mmData
        
path = '/media/sf_Python/data/from_fileprep/'
filename = 'F_ES_19981222to20160204signals.csv'
ticker = filename[:4]
#signal = 'ZZ 0.02,-0.005'
signal = 'volCheck'
#for matching columns in directional filter
dropCol = ['Open','High','Low','Close','Volume','gainAhead','signal','dates']
#            'Pri_RSI_Y1','Pri_RSI_Y2','Pri_RSI_Y3','Pri_RSI_Y4']

RSILookback = 1.5
zScoreLookback = 10
ATRLookback = 5
DPOLookback = 3
# relATR -> 10 zscore atr lookback + shift 5 + 10zscore atr rel lookback
maxlb = max(RSILookback, zScoreLookback, ATRLookback, DPOLookback, 60)

start_time = time.time()
model_metrics = init_report()

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
RFE_estimator = [ 
        ("GradientBoostingRegressor",GradientBoostingRegressor()),\
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

dataSet['gainAhead'] = gainAhead(dataSet.Close)
dataSet['runsScore10'] = runsZScore(dataSet.gainAhead,10)
dataSet['percentUpDays10'] = percentUpDays(dataSet.gainAhead,10)
dataSet['mean60_ga'] = pd.rolling_mean(dataSet.gainAhead,60)
dataSet['std60_ga'] = pd.rolling_std(dataSet.gainAhead,60)
dataSet['kurt60_ga'] = pd.rolling_kurt(dataSet.gainAhead,60)
dataSet['skew60_ga'] = pd.rolling_skew(dataSet.gainAhead,60)
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

dataSet = dataSet.dropna()

#toxCheck = ['TOX20', 'TOX25']
#toxCheck = ['TOX25']
tox_adj_proportion = 1 # proportion of # -1's / #1's 

DD95_limit = 0.20
initial_equity = 1.0
#dataLoadStartDate = "1998-12-22"
#dataLoadEndDate = "2016-01-04"
filterName = 'VF1'
testFirstYear = "1999-01-01"
testFinalYear = "2015-12-31"
validationFirstYear ="2016-01-01"
validationFinalYear ="2016-02-05"

test_split = 0.33 #test split
iterations = 1 #for sss
tox_adj_proportion = 0
feature_selection = 'None'
nfeatures = 35


wfStep=1
minDataPoints = 400
cutoff = 25 #%
tox_adj_proportion = 1
signal  = 'TOX'+str(cutoff)

#begin short system
# remove long trades
#wf_is_periods = [500]
volatilityFilter = {}
#set validation period
if dataSet.ix[:testFinalYear].shape[0] < minDataPoints:
    print 'Not enough datapoints. Adjusting testFinalYear to', dataSet.ix[minDataPoints:].index[-1]
    volatilityFilter[filterName+signal] = dataSet.ix[minDataPoints:]
    initialInSamplePeriod = minDataPoints
else:
    volatilityFilter[filterName+signal] = dataSet.ix[validationFirstYear:]
    initialInSamplePeriod = dataSet.ix[:validationFirstYear].shape[0]


#for sst in sstDict:
#    volatilityFilter[sst] = pd.concat([sstDict[sst][sstDict[sst].signals == -1]\
#        .signals, dataSet],axis=1,join='inner').drop(['signals'],axis=1)
#    volatilityFilter[sst].index.name = 'dates'

#dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']
dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','dates']
#model_metrics = init_report()       
sstDictVF = {}
#for tc in toxCheck:
for i,m in enumerate(models):
    #for wfStep in wfSteps:
    for vfDataSet in volatilityFilter:
    
        testFirstYear_ss = dataSet.index[0]
        #index -1 would be index[0](validation start) on the dataset so -2 would be day before validation
        testFinalYear_ss = dataSet.ix[:volatilityFilter[vfDataSet].index[0]].index[-2]
        validationFirstYear_ss = volatilityFilter[vfDataSet].index[0]
        validationFinalYear = volatilityFilter[vfDataSet].index[-1]
        #for wf_is_period in wf_is_periods:
        #    #check
        #    nrows_is = volatilityFilter[vfDataSet].ix[:testFinalYear_ss].dropna().shape[0]
        #    if wf_is_period > nrows_is:
        #        print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
        #        print 'Adjusting to', testFinalYear_ss, 'to', volatilityFilter[vfDataSet].iloc[wf_is_period].name
        #        testFirstYear_ss = volatilityFilter[vfDataSet].iloc[0].name
        #        testFinalYear_ss = volatilityFilter[vfDataSet].iloc[wf_is_period].name
        #        validationFirstYear_ss = volatilityFilter[vfDataSet].iloc[wf_is_period+1].name
        #        
        metaData = {'ticker':ticker, 't_start':testFirstYear_ss, 't_end':testFinalYear_ss,\
                 'signal':signal, 'data_type':vfDataSet,\
                 'test_split':0, 'iters':1, 'tox_adj':tox_adj_proportion,\
                 'n_features':nfeatures, 'FS':feature_selection,'rfe_model':RFE_estimator[0],\
                 'v_start':validationFirstYear_ss, 'v_end':validationFinalYear,'wf_step':wfStep\
                  }
        #reset index for integer index
        mmData_v = volatilityFilter[vfDataSet].reset_index()
        nrows_oos = mmData_v.shape[0]
        #datay_signal = mmData_v[['signal', 'prior_index']]
        datay_gainAhead = mmData_v.gainAhead
        cols = mmData_v.drop(dropCol, axis=1).columns.shape[0]
        metaData['cols']=cols
        feature_names = []
        print '\nTotal %i features: ' % cols
        for i,x in enumerate(mmData_v.drop(dropCol, axis=1).columns):
            print i,x+',',
            feature_names = feature_names+[x]
        if nfeatures > cols:
            print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
            print 'Adjusting to', cols, 'features..'
            nfeatures = cols  
        if feature_selection == 'None':
            nfeatures = cols 
        print '\n\nNew WF train/predict loop for', m[1]
        #print "\nStarting Walk Forward run on filtered data..."
        if feature_selection == 'Univariate':
            print "Using top %i %s features" % (nfeatures, feature_selection)
        else:
            print "Using %s features" % feature_selection
        
        print "%i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (minDataPoints, nrows_oos,wfStep)
        
        #cm_y_train = np.array([])
        cm_y_test = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        for j in range(0,len(volatilityFilter[vfDataSet])):
            #print j,
            #set last date of training data
            testFinalYear = dataSet.ix[:volatilityFilter[vfDataSet].index[j]].index[-2]
            
            mmData = dataSet.ix[:testFinalYear].reset_index()
            toxCutoff = getToxCutoff2(abs(mmData.gainAhead),cutoff)
            print testFinalYear, 'Cutoff: ', toxCutoff[1], 'GA', mmData_v.gainAhead.iloc[j],
            #remove datapoints that dont meet cutoff
            #mmData_adj = mmData[abs(mmData.gainAhead) >=toxCutoff[1]]
            #relabel signals. 
            toxSignals = pd.Series(data=np.where(abs(mmData.gainAhead) >=toxCutoff[1],1,-1), name='signal',index=mmData.index)
            
            mmData_adj = adjustDataProportion2(pd.concat([mmData,toxSignals],axis=1), tox_adj_proportion,verbose=False)
            
            dataX = mmData_adj.drop(dropCol+['signal'], axis=1)
            #print 'testFinalYear',testFinalYear,'validationFirstYear', mmData_v.iloc[j].name
                            
            #dy = np.zeros_like(datay_gainAhead.iloc[j])
            #dX = np.zeros_like(dataX)
            X_test = mmData_v.drop(dropCol, axis=1).iloc[j].values
            y_test = np.where(mmData_v.gainAhead.iloc[j]>toxCutoff[1],1,-1)
            X_train = dataX.values
            y_train = mmData_adj.signal.values
            #    runName = 'ShortDataSet_'+'vf_' +vf_name +' df_' + m[0]+'_is'+str(wf_is_period)+'_oos'+str(wfStep)+vfDataSet
            #    model_metrics, sstDictVF[runName] = wf_regress_validate(unfilteredData, volatilityFilter[vfDataSet], [m], model_metrics, wf_is_period, metaData, showPDFCDF=0)
            
            if feature_selection != 'None':
                if feature_selection == 'RFECV':
                    #Recursive feature elimination with cross-validation: 
                    #A recursive feature elimination example with automatic tuning of the
                    #number of features selected with cross-validation.
                    rfe = RFECV(estimator=RFE_estimator, step=1)
                    rfe.fit(X_train, y_train)
                    #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
                    featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
                    print 'Top %i RFECV features' % len(featureRank)
                    print featureRank    
                    metaData['featureRank'] = str(featureRank)
                    X_train = rfe.transform(X_train)
                    X_test = rfe.transform(X_test.reshape(1, -1))
                else:
                    #Univariate feature selection
                    skb = SelectKBest(chi2, k=nfeatures)
                    skb.fit(X_train, y_train)
                    #dX_all = np.vstack((X_train.values, X_test.values))
                    #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
                    #dX_v_rfe = X_new[dX_t.shape[0]:]
                    X_train = skb.transform(X_train)
                    X_test = skb.transform(X_test.reshape(1, -1))
                    featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
                    metaData['featureRank'] = str(featureRank)
                    #print 'Top %i univariate features' % len(featureRank)
                    #print featureRank
            else:
                nfeatures = cols
                
            m[1].fit(X_train, y_train)
            #trained_models[m[0]] = pickle.dumps(m[1])
                        
            #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))              
            y_pred_oos = m[1].predict(X_test.reshape(1,-1))
            print 'ytrue',y_test,'ypred', y_pred_oos
            
            if m[0][:2] == 'GA':
                print featureRank
                print '\nProgram:', m[1]._program
                #print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
            
            #cm_y_train = np.concatenate([cm_y_train,y_train])
            cm_y_test = np.hstack([cm_y_test,y_test])
            #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
            cm_y_pred_oos = np.hstack([cm_y_pred_oos,y_pred_oos])
            #cm_train_index = np.concatenate([cm_train_index,train_index])
            cm_test_index = np.hstack([cm_test_index,j])
            
        #create signals 1 and -1
        #cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos])
        #cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test])
        oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                            ticker, validationFirstYear_ss, validationFinalYear, iterations, 'VF',1)
        
        #data is filtered so need to fill in the holes. signal = 0 for days that filtered
        st_oos_filt= pd.DataFrame()
        st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
        #st_oos_filt['prior_index'] = pd.Series(cm_y_pred_oos)
        #st_oos_filt['gainAhead'] =  datay_gainAhead[cm_test_index].reset_index().gainAhead
        st_oos_filt.index = mmData_v['dates'].iloc[cm_test_index]
        st_oos_filt = pd.concat([st_oos_filt,dataSet.gainAhead], axis=1, join='outer').ix[validationFirstYear:validationFinalYear]
        st_oos_filt['signals'].fillna(0, inplace=True)
        '''
        #change 0's to -1 for confusion matrix, input signal is -1, so 0->1
        cm_y_pred_oos = np.where(st_oos_filt['signals'].values==0,1,st_oos_filt['signals'].values)
        cm_y_test = np.where(st_oos_filt.gainAhead>0,1,-1)
        datay_gainAhead = st_oos_filt.gainAhead
        cmatrix_test_index = st_oos_filt.reset_index().index
        
        #plot learning curve, knn insufficient neighbors
        #if showLearningCurve:
        #    try:
        #        plot_learning_curve(m[1], m[0], X_train,y_train, scoring='r2')        
        #    except:
        #        pass
            
        #plot out-of-sample data
        plt.figure()
        coef, b = np.polyfit(cm_y_pred_oos, cm_y_test, 1)
        plt.title('Out-of-Sample')
        plt.ylabel('gainAhead')
        plt.xlabel('ypred gainAhead')
        plt.plot(cm_y_pred_oos, cm_y_test, '.')
        plt.plot(cm_y_pred_oos, coef*cm_y_pred_oos + b, '-')
        plt.show()
        '''
        print 'Metrics for', filterName
        oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1], ticker, validationFirstYear, validationFinalYear, iterations, signal,1)
        #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int), close, 'LONG', 1)
        #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int), close, 'SHORT', -1)
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead, cm_test_index, m, metaData)
        #model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead, cm_test_index, m, metaData)
        
        sstDictVF[m[0]+vfDataSet]=st_oos_filt
        
#find and save best volatility model
scored_models = model_metrics.loc[model_metrics['sample'] == filterName].reset_index()
#zscore
scored_models['f1mm'] =minmax_scale(scored_models.f1.reshape(-1, 1))
scored_models['accmm'] =minmax_scale(scored_models.acc.reshape(-1, 1))
scored_models['precmm'] =minmax_scale(scored_models.prec.reshape(-1, 1))
scored_models['recmm'] = minmax_scale(scored_models.rec.reshape(-1, 1))
scored_models['fn_magmm'] =-minmax_scale(scored_models.fn_mag.reshape(-1, 1))
scored_models['fp_magmm'] =-minmax_scale(scored_models.fp_mag.reshape(-1, 1))
scored_models['scoremm'] =  scored_models.f1mm+scored_models.accmm+scored_models.precmm+\
                                scored_models.recmm+scored_models.fn_magmm+scored_models.fp_magmm

#softmax
scored_models['f1sm'] = softmax_score(scored_models['f1'])
scored_models['accsm'] = softmax_score(scored_models['acc'])
scored_models['precsm'] = softmax_score(scored_models['prec'])
scored_models['recsm'] = softmax_score(scored_models['rec'])
scored_models['fn_magsm'] =-softmax_score(minmax_scale(scored_models['fn_mag']))
scored_models['fp_magsm'] =-softmax_score(minmax_scale(scored_models['fp_mag']))
scored_models['scoresm'] =  scored_models.f1sm+scored_models.accsm\
            +scored_models.precsm+scored_models.recsm+scored_models.fn_magsm+scored_models.fp_magsm

#forgot why i multiplied sm and mm, but it probably had to do with some balancing
scored_models['final_score'] = scored_models['scoresm'] * scored_models['scoremm']
scored_models = scored_models.sort_values(['final_score'], ascending=False)
bestModel = scored_models.iloc[0]

#save all sst
for m in models:
    if bestModel['params'] == str(m[1]):
        print  '\n\nBest model found...\n', m[1]
        bm = m[1]
print 'Number of features: ', bestModel.n_features, bestModel.FS
print 'WF Signal:', bestModel.signal
print 'WF Initial In-Sample Period:', initialInSamplePeriod

sstDictVF_ = copy.deepcopy(sstDictVF)

for runName in sstDictVF:

    compareEquity(sstDictVF[runName],runName)
    '''
    filename_sst = ticker + '_' +runName+ '_' +\
                   validationFirstYear_ss + 'to' + validationFinalYear + '_' +\
                   re.sub(r'[^\w]', '', str(datetime.today()))[:14] + '.csv'
    sstDictVF[runName].drop(['prior_index'],axis=1).to_csv(save_path_rs+'SST_'+filename_sst)    
    
    sstDictVF[runName].signals[sstDictVF[runName].signals ==0] = 1
    compareEquity(sstDictVF[runName],'Combined Longs'+runName)
    #print sstDictVF2[runName].head()
    sstDictVF2[runName].signals[sstDictVF2[runName].signals ==1] = -2
    #print sstDictVF2[runName].head()
    sstDictVF2[runName].signals[sstDictVF2[runName].signals ==0] = 1
    #print sstDictVF2[runName].head()
    sstDictVF2[runName].signals[sstDictVF2[runName].signals ==-2] = 0
    #print sstDictVF2[runName].head()
    compareEquity(sstDictVF2[runName],'Flat Longs'+runName)
    '''
