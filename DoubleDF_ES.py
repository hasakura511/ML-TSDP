# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:08:04 2015
for debugging
[x for x in newfile.ix[oldfile.index[0]:].index if x not in oldfile.index]
"""
import math
import random
import numpy as np
import pandas as pd
import copy
import string
#import Quandl
import pickle
import re
import talib as ta
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve,\
                         directional_scoring, compareEquity, displayRankedCharts
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25,\
                            maxCAR25, wf_regress_validate, sss_regress_train,\
                            calcEquity_df, createYearlyStats, CAR25_df,\
                            createBenchmark, adjustDataProportion2
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, getToxCutoff2
from sklearn.preprocessing import scale, robust_scale, minmax_scale
from scipy import stats
from datetime import datetime as dt
import datetime
import time
import numpy as np
import pandas as pd
import sklearn

#from pandas_datareader import DataReader
from sklearn.externals import joblib
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

#from sklearn.neural_network import MLPClassifier

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer

from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from IPython.display import Image
#import pydot

#from sknn.mlp import Classifier, Layer
#from sknn.backend import lasagne
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFE, RFECV
systemName = '2DF_ES_'
file_path = '/media/sf_Python/data/from_vf_to_df/'
model_path = '/media/sf_Python/saved_models/DF/'
save_path_sst = '/media/sf_Python/data/from_df_to_dps/'
#save best models for regime switching here
save_path_rs = '/media/sf_Python/data/to_RS/'
scoredModelPath = '/media/sf_Python/FDR_'
#data_type = 'HV'
#filteredDataFile = 'HV_F_ES_TOX25_VotingHard_2001-01-11to2015-12-31_20160107004343.csv'
#data_type = 'LV'
#filteredDataFile = 'LV_F_ES_TOX25_VotingHard_2001-01-11to2015-12-31_20160107004343.csv'
data_type = 'ALL'
filteredDataFile = 'OHLCV_F_ES_TOX25_VotingHard_v2015-01-01to2017-01-01_20160214201951.csv'
unfilteredDataFile = 'OHLCV_F_ES_TOX25_VotingHard_v2015-01-01to2017-01-01_20160214201951.csv'
vf_name = 'None'
ticker = 'F_ES'
feature_selection = 'Univariate' #RFECV OR Univariate
#vmodel_file = 'VF_20151215193820622423GBCmodelGBCsignalTOX250011497tox_adj1date_end20150331tickerF_SPdate__start20090101Name27dtypeobject.pkl'
signal = 'LongGT0'
DD95_limit = 0.20
initial_equity = 100000.0
#DF1
wfSteps=[1]
wf_is_periods = [250]
testFirstYear = "1999-01-01"
testFinalYear = "2002-12-31"
validationFirstYear ="2003-01-01"
validationFinalYear ="2016-02-05"

test_split = 0.33 #test split
iterations = 1 #for sss
tox_adj_proportion = 0
nfeatures = 34

#charts to display in summary
numCharts = 2

#number of sst of systems to save for DPS/RS
nTopSystems = 'ALL'

if dt.strptime(testFinalYear, '%Y-%m-%d') >= dt.strptime(validationFirstYear, '%Y-%m-%d'):
    raise ValueError, 'testFinalYear >= validationFirstYear'
    
#RSILookback = 1.5
#zScoreLookback = 20
#ATRLookback = 5
#beLongThreshold = 0.0
#DPOLookback = 3


start_time = time.time()
equityCurves = {}
model_metrics = init_report()
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
          #                             n_jobs=1, verbose=0, random_state=None)),\
         #("GA_Reg2", SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, comparison=True, transformer=False, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=1, verbose=0, parsimony_coefficient=0.01, random_state=0)),\
        #("RandomForestRegressor",RandomForestRegressor()),\
        #("AdaBoostRegressor",AdaBoostRegressor()),\
        #("BaggingRegressor",BaggingRegressor()),\
        #("_ET",ExtraTreesRegressor()),\
        #("GradientBoostingRegressor",GradientBoostingRegressor()),\
        #("IsotonicRegression",IsotonicRegression()),\ #ValueError: X should be a 1d array
        #("KernelRidge",KernelRidge()),\
        #("DecisionTreeRegressor",DecisionTreeRegressor()),\
        #("ExtraTreeRegressor",ExtraTreeRegressor()),\
        #("ARDRegression", ARDRegression()),\
        #("LogisticRegression", LogisticRegression()),\ # ValueError("Unknown label type: %r" % y)
        #("Ridge",Ridge()),\
        #("RANSACRegressor",RANSACRegressor()),\
        #("LinearRegression",LinearRegression()),\
        #("Lasso",Lasso()),\
        #("LassoLars",LassoLars()),\
        #("BayesianRidge", BayesianRidge()),\
        #("PassiveAggressiveRegressor",PassiveAggressiveRegressor()),\ #all zero array
        #("SGDRegressor",SGDRegressor()),\
        #("TheilSenRegressor",TheilSenRegressor()),\
        #("_KNNu", KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)),\
        ("KNNd", KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)),\
        #("KNeighborsRegressor-u,p1,n15", KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=1, metric='minkowski', metric_params=None, n_jobs=-1)),\
        #("RadiusNeighborsRegressor",RadiusNeighborsRegressor()),\
        #("LinearSVR", LinearSVR()),\
        #("rbf1SVR",SVR(kernel='rbf', C=1, gamma=0.1)),\
        #("rbf10SVR",SVR(kernel='rbf', C=10, gamma=0.1)),\
        #("rbf100SVR",SVR(kernel='rbf', C=1e3, gamma=0.1)),\
        #("polySVR",SVR(kernel='poly', C=1e3, degree=2)),\
         ]

vfData = pd.read_csv(file_path+filteredDataFile, index_col='dates')
vfData[signal] = np.where(vfData.gainAhead>0,1,-1)

qt = pd.read_csv(file_path+unfilteredDataFile, index_col='dates')


if data_type == 'ALL':
    #OHLC data (all)
    unfilteredData = create_indicators(qt).dropna()
    #inner join vfData
    filteredData = pd.concat([vfData,unfilteredData], axis=1, join='inner').dropna()
    unfilteredData['gainAhead'] = qt['gainAhead'] #add gainAdhead after merge. This will be used for CAR25
    dataSet = pd.concat([filteredData[' OPEN'],filteredData[' HIGH'],filteredData[' LOW'],filteredData[' CLOSE'],filteredData[' VOL'],pd.Series(unfilteredData.reset_index().index, index=unfilteredData.index)], axis=1)
else:
    #HV/LV
    unfilteredData = create_indicators(qt)   
    #inner join vfData
    filteredData = pd.concat([vfData,unfilteredData], axis=1, join='inner').dropna()
    unfilteredData['gainAhead'] = qt['gainAhead'] #add gainAdhead after merge. This will be used for CAR25
    dataSet = pd.concat([filteredData[' OPEN'],filteredData[' HIGH'],filteredData[' LOW'],filteredData[' CLOSE'],filteredData[' VOL'], filteredData['Unnamed: 0']], axis=1)

dataSet.columns = ['Open','High','Low','Close','Volume','prior_index']
close = unfilteredData.reset_index().Close
nrows = dataSet.shape[0]
print "Successfully loaded %i rows" % nrows

#no shifting allowed here because it is filtered and data has gaps
#choose features for modeling
dataSet['Pri_RSI'] = filteredData['Pri_RSI']
dataSet['Pri_RSI_Y1'] = filteredData['Pri_RSI_Y1']
dataSet['Pri_RSI_Y2'] = filteredData['Pri_RSI_Y2']
dataSet['Pri_RSI_Y3'] = filteredData['Pri_RSI_Y3']
dataSet['Pri_RSI_Y4'] = filteredData['Pri_RSI_Y4']
dataSet['Pri_ATR'] = filteredData['Pri_ATR']
dataSet['Pri_ATR_Y1'] =filteredData['Pri_ATR_Y1']
dataSet['Pri_ATR_Y2'] =filteredData['Pri_ATR_Y2']
dataSet['Pri_ATR_Y3'] = filteredData['Pri_ATR_Y3']
dataSet['Rel_ATR'] = filteredData['Rel_ATR']
dataSet['priceChange'] = filteredData['priceChange']
dataSet['priceChangeY1'] = filteredData['priceChangeY1']
dataSet['priceChangeY2'] = filteredData['priceChangeY2']
dataSet['priceChangeY3'] = filteredData['priceChangeY3']
dataSet['pctChangeOpen'] = filteredData['pctChangeOpen']
dataSet['pctChangeLow'] = filteredData['pctChangeLow']
dataSet['pctChangeHigh'] = filteredData['pctChangeHigh']
dataSet['Pri_DPO'] = filteredData['Pri_DPO']
dataSet['GARCH'] = filteredData['GARCH']
dataSet['GARCH_Y1'] = filteredData['GARCH_Y1']
dataSet['GARCH_Y2'] = filteredData['GARCH_Y2']
dataSet['autoCor'] = filteredData['autoCor']
dataSet['autoCor_Y1'] =filteredData['autoCor_Y1']
dataSet['autoCor_Y2'] = filteredData['autoCor_Y2']
dataSet['autoCor_Y3'] = filteredData['autoCor_Y3']
dataSet['K_Eff'] = filteredData['K_Eff']
dataSet['K_Eff_Y1'] = filteredData['K_Eff_Y1']
dataSet['K_Eff_Y2'] = filteredData['K_Eff_Y2']
dataSet['K_Eff_Y3'] = filteredData['K_Eff_Y3']
dataSet['volSpike'] = filteredData['volSpike']
dataSet['volSpike_Y1'] = filteredData['volSpike_Y1']
dataSet['volSpike_Y2'] = filteredData['volSpike_Y2']
dataSet['volSpike_Y3'] = filteredData['volSpike_Y3']
#dataSet['F_ED_zScore'] = filteredData['F_ED_zScore']
#dataSet['F_ED_zScore_Y1'] = filteredData['F_ED_zScore_Y1']
#dataSet['F_ED_zScore_Y2'] = filteredData['F_ED_zScore_Y2']
#dataSet['F_ED_zScore_Y3'] = filteredData['F_ED_zScore_Y3']
#dataSet['F_GC_zScore'] = filteredData['F_GC_zScore']
#dataSet['F_GC_zScore_Y1'] = filteredData['F_GC_zScore_Y1']
#dataSet['F_GC_zScore_Y2'] = filteredData['F_GC_zScore_Y2']
#dataSet['F_GC_zScore_Y3'] = filteredData['F_GC_zScore_Y3']
#dataSet['F_CL_zScore'] = filteredData['F_CL_zScore']
#dataSet['F_CL_zScore_Y1'] = filteredData['F_CL_zScore_Y1']
#dataSet['F_CL_zScore_Y2'] = filteredData['F_CL_zScore_Y2']
#dataSet['F_CL_zScore_Y3'] = filteredData['F_CL_zScore_Y3']
#dataSet['F_DX_zScore'] = filteredData['F_DX_zScore']
#dataSet['F_DX_zScore_Y1'] = filteredData['F_DX_zScore_Y1']
#dataSet['F_DX_zScore_Y2'] = filteredData['F_DX_zScore_Y2']
#dataSet['F_DX_zScore_Y3'] = filteredData['F_DX_zScore_Y3']

dataSet['F_ED_WOWzScore'] = filteredData['F_ED_WOWzScore']
dataSet['F_GC_WOWzScore'] = filteredData['F_GC_WOWzScore']
dataSet['F_CL_WOWzScore'] = filteredData['F_CL_WOWzScore']
dataSet['F_DX_WOWzScore'] = filteredData['F_DX_WOWzScore']
dataSet['F_US_WOWzScore'] = filteredData['F_US_WOWzScore']
#add sector spdrs								
dataSet['S_XLB_WOWzScore'] = filteredData['S_XLB_WOWzScore']
dataSet['S_XLE_WOWzScore'] = filteredData['S_XLE_WOWzScore']
dataSet['S_XLY_WOWzScore'] = filteredData['S_XLY_WOWzScore']
dataSet['S_XLF_WOWzScore'] = filteredData['S_XLF_WOWzScore']
dataSet['S_XLU_WOWzScore'] = filteredData['S_XLU_WOWzScore']
dataSet['S_XLP_WOWzScore'] = filteredData['S_XLP_WOWzScore']
dataSet['S_XLV_WOWzScore'] = filteredData['S_XLV_WOWzScore']
dataSet['S_XLK_WOWzScore'] = filteredData['S_XLK_WOWzScore']
dataSet['S_XLI_WOWzScore'] = filteredData['S_XLI_WOWzScore']
#add china
dataSet['S_HSI_WOWzScore'] = filteredData['S_HSI_WOWzScore']
dataSet['S_SEC_WOWzScore'] = filteredData['S_SEC_WOWzScore']
#for col in filteredData:
#    if col[:2] =='S_' and col[-2] !='Y' and len(col)>5:
#        dataSet[col] = filteredData[col]

#dataSet['runsScore10'] = filteredData['runsScore10']
#dataSet['percentUpDays10'] = filteredData['percentUpDays10']
dataSet['mean60_ga'] = filteredData['mean60_ga']
dataSet['std60_ga'] = filteredData['std60_ga']
#dataSet['kurt60_ga'] = filteredData['kurt60_ga']
#dataSet['skew60_ga'] = filteredData['skew60_ga']

#transformed relative to close
#PRICE LEVEL
dataSet['linreg']=filteredData['linreg']
dataSet['tsf']=filteredData['tsf']
dataSet['support'] =filteredData['support']
dataSet['resistance']=filteredData['resistance']
dataSet['bbupper']=filteredData['bbupper']
dataSet['bbmiddle']=filteredData['bbmiddle']
dataSet['bblower']=filteredData['bblower']

#transformed to minmax
#MOMENTUM
dataSet['roc']=filteredData['roc']
dataSet['rocp']=filteredData['rocp']
dataSet['rocr']=filteredData['rocr']
dataSet['rocr100']=filteredData['rocr100']
dataSet['trix'] =filteredData['trix']
dataSet['ultosc'] =filteredData['ultosc']
dataSet['willr']=filteredData['willr']

# transformed to zs
#MOMENTUM
dataSet['linreg_slope']=filteredData['linreg_slope']
dataSet['adx']=filteredData['adx']
dataSet['adxr']=filteredData['adxr']
dataSet['macd']=filteredData['macd']
dataSet['macdsignal']=filteredData['macdsignal']
dataSet['macdhist']=filteredData['macdhist']
dataSet['minus_di']=filteredData['minus_di']
dataSet['minus_dm']=filteredData['minus_dm']
dataSet['cmo']=filteredData['cmo']
dataSet['mom']=filteredData['mom']
dataSet['plus_di']=filteredData['plus_di']
dataSet['plus_dm']=filteredData['plus_dm']
dataSet['ppo']=filteredData['ppo']
#Volume
#dataSet['ad']=filteredData['ad']
#dataSet['adosc']=filteredData['adosc']
dataSet['obv']=filteredData['obv']
#VOLATILITY
dataSet['atr']=filteredData['atr']
dataSet['natr']=filteredData['natr']
dataSet['trange']=filteredData['trange']
#CYCLE
dataSet['ht_dcperiod']=filteredData['ht_dcperiod']
dataSet['ht_dcphase']=filteredData['ht_dcphase']
dataSet['inphase']=filteredData['inphase']
dataSet['quadrature']=filteredData['quadrature']

# div 100
#MOMENTUM
dataSet['apo']=filteredData['apo']
dataSet['aroondown']=filteredData['aroondown']
dataSet['aroonup']=filteredData['aroonup']
dataSet['aroonosc']=filteredData['aroonosc']
dataSet['cci']=filteredData['cci']
dataSet['dx']=filteredData['dx']
dataSet['mfi']=filteredData['mfi']
dataSet['rsi']=filteredData['rsi']
dataSet['slowk']=filteredData['slowk']
dataSet['slowd']=filteredData['slowd']
dataSet['fastk']=filteredData['fastk']
dataSet['fastd']=filteredData['fastd']
dataSet['rsifastk']=filteredData['rsifastk']
dataSet['rsifastd']=filteredData['rsifastd']

#no transform
#MOMENTUM
dataSet['bop']=filteredData['bop']
#CYCLE
dataSet['sine']=filteredData['sine']
dataSet['leadsine']=filteredData['leadsine']
dataSet['ht_trendmode']=filteredData['ht_trendmode']

#turn this on to see if the algo can detect the cheat.
#dataSet['Cheat'] = filteredData['gainAhead']

dataSet['gainAhead'] = filteredData['gainAhead']

dataSet['signal'] = filteredData[signal]

#check dataSet for nan/inf beyond initial lookback periods maxlb
for col in dataSet:
    if sum(np.isnan(dataSet[col].values))>0:
        print dataSet[col][np.isnan(dataSet[col].values)]
        raise ValueError, 'nan in %s' % col
    elif sum(np.isinf(dataSet[col].values))>0:
        print dataSet[col][np.isnan(dataSet[col].values)]
        raise ValueError, 'inf in %s' % col
    elif sum(np.isneginf(dataSet[col].values))>0:
        print dataSet[col][np.isnan(dataSet[col].values)]
        raise ValueError, '-inf in %s' % col
        
#adjust validation Final Year to last date in dataSet
if dataSet.index.to_datetime()[-1] < dt.strptime(validationFinalYear,'%Y-%m-%d'):
    validationFinalYear = dataSet.index[-1]

#begin first filter. walk forward validation
sstDict = {}
filterName = 'DF1'
input_signal = 1
longMemory = False

for i,m in enumerate(models):
    for wfStep in wfSteps:
        for wf_is_period in wf_is_periods:
        
            #check
            nrows_is = dataSet.ix[:testFinalYear].dropna().shape[0]
            if wf_is_period > nrows_is:
                print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
                print 'Adjusting to', nrows_is, 'rows..'
                wf_is_period = nrows_is
                
            metaData = {'ticker':ticker, 't_start':testFirstYear, 't_end':testFinalYear,\
                     'signal':signal, 'data_type':data_type,'filter':filterName, 'input_signal':input_signal,\
                     'test_split':0, 'iters':1, 'tox_adj':tox_adj_proportion,'longMemory':longMemory,\
                     'n_features':nfeatures, 'FS':feature_selection,'rfe_model':RFE_estimator[0],\
                     'v_start':validationFirstYear, 'v_end':validationFinalYear,'wf_step':wfStep\
                      }
            runName = data_type+'_'+filterName+'_' + m[0]+'_i'+str(wf_is_period)#+'_o'+str(wfStep)
            model_metrics, sstDict[runName] = wf_regress_validate(unfilteredData, dataSet, [m], model_metrics,\
                                                wf_is_period, metaData, showPDFCDF=1,longMemory=longMemory)

scored_models, bestModel = directional_scoring(model_metrics,filterName)

#display best model
for m in models:
    if bestModel['params'] == str(m[1]):
        print  '\n\nBest model found...\n', m[1]
        bm = m[1]
print 'Number of features: ', bestModel.n_features, bestModel.FS
print 'WF In-Sample Period:', bestModel.rows
print 'WF Out-of-Sample Period:', bestModel.wf_step
print 'Long Memory: ', longMemory
DF1_BMrunName = data_type+'_'+filterName+'_'  + bestModel.model + '_i'+str(bestModel.rows)#+'_o'+str(bestModel.wf_step)
compareEquity(sstDict[DF1_BMrunName].ix[validationFirstYear:validationFinalYear],runName)

#display CAR25 of best model
#CAR25_L1_oos = CAR25(signal, sst.signals, sst.prior_index.astype(int), close, 'LONG', 1)
#CAR25_Sn1_oos = CAR25(signal, sst.signals, sst.prior_index.astype(int), close, 'SHORT', -1)  

#save best model sst file
#filename_sst = ticker + '_' +runName+ '_' +\
#                   validationFirstYear + 'to' + validationFinalYear + '_' +\
#                   re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
#sst.drop(['prior_index'],axis=1).to_csv(save_path_rs+'SST_BM_DF1_'+filename_sst)


'''        
#save scored model summary
filename_fpi =data_type +'_'+systemName + '_' +\
                   validationFirstYear + 'to' + validationFinalYear + '_' +\
                   re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'                   
scored_models.to_csv(scoredModelPath+filterName+filename_fpi)
'''


############################################
#begin short system
        
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
         ("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
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
         #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=5, weights='distance')),\
         #uniform is faster
         #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
         #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
         #("VotingHard", VotingClassifier(estimators=[\
         #    ("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
         #    ("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
         #    ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
         #    ("QDA", QuadraticDiscriminantAnalysis()),\
         #    ("GNBayes",GaussianNB()),\
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
             #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
         #    ("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
         #       ], voting='hard', weights=None)),
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
         
#toxCheck = ['TOX20', 'TOX25']
#toxCheck = ['TOX25']
tox_adj_proportion = 1 # proportion of # -1's / #1's 

#dataLoadStartDate = "1998-12-22"
#dataLoadEndDate = "2016-01-04"
filterName = 'DF2'
feature_selection = 'None'
longMemory =  True

wfStep=1
initialInSamplePeriod = 400
cutoff = 25 #%
tox_adj_proportion = 1
signal  = 'TOX'+str(cutoff)
input_signal =-1
#begin short system
# remove long trades
#wf_is_periods = [500]
#get -1's from DF2. also includes -1's from DF1 because 0's filled in as -1.
sstDF3 = {}
for sst in sstDict:
    sstDF3['-1_400(250)'] = pd.concat([sstDict[sst][sstDict[sst].signals == input_signal]\
        .signals, dataSet],axis=1,join='inner').drop(['signals'],axis=1)
    sstDF3['-1_400(250)'].index.name = 'dates'
    


#for sst in sstDict:
#    sstDF3[sst] = pd.concat([sstDict[sst][sstDict[sst].signals == -1]\
#        .signals, dataSet],axis=1,join='inner').drop(['signals'],axis=1)
#    sstDF3[sst].index.name = 'dates'

dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']
dropCol_v = ['Open','High','Low','Close', 'Volume','gainAhead','signal', 'prior_index']
#model_metrics = init_report()       
model_metrics = init_report()
sstDictVF = {}
#for tc in toxCheck:
for i,m in enumerate(models):
    #for wfStep in wfSteps:
    for sst in sstDF3:
        testFirstYear_ss=testFirstYear
        testFinalYear_ss = testFinalYear
        validationFirstYear_ss = validationFirstYear
        nrows_is = sstDF3[sst].ix[:testFinalYear_ss].dropna().shape[0]
        if initialInSamplePeriod > nrows_is:
            print 'Walkforward insample period of', initialInSamplePeriod, 'is greater than in-sample data of ', nrows_is, '!'
            print 'Adjusting to', testFinalYear_ss, 'to', sstDF3[sst].iloc[initialInSamplePeriod].name
            testFirstYear_ss = sstDF3[sst].iloc[0].name
            testFinalYear_ss = sstDF3[sst].iloc[initialInSamplePeriod].name
            validationFirstYear_ss = sstDF3[sst].iloc[initialInSamplePeriod+1].name
        #for initialInSamplePeriod in wf_is_periods:
        #    #check
        #    nrows_is = sstDF3[sst].ix[:testFinalYear_ss].dropna().shape[0]
        #    if initialInSamplePeriod > nrows_is:
        #        print 'Walkforward insample period of', initialInSamplePeriod, 'is greater than in-sample data of ', nrows_is, '!'
        #        print 'Adjusting to', testFinalYear_ss, 'to', sstDF3[sst].iloc[initialInSamplePeriod].name
        #        testFirstYear_ss = sstDF3[sst].iloc[0].name
        #        testFinalYear_ss = sstDF3[sst].iloc[initialInSamplePeriod].name
        #        validationFirstYear_ss = sstDF3[sst].iloc[initialInSamplePeriod+1].name
        #

        #reset index for integer index
        mmData_v = sstDF3[sst].ix[validationFirstYear_ss:]
        nrows_oos = mmData_v.shape[0]
        #datay_signal = mmData_v[['signal', 'prior_index']]
        datay_gainAhead = mmData_v.gainAhead
        cols = mmData_v.drop(dropCol_v, axis=1).columns.shape[0]
        metaData['cols']=cols
        feature_names = []
        print '\nTotal %i features: ' % cols
        for i,x in enumerate(mmData_v.drop(dropCol_v, axis=1).columns):
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
            print "Feature Selection: %s Using %i features" % (feature_selection, nfeatures)
        
        metaData = {'ticker':ticker, 't_start':testFirstYear_ss, 't_end':testFinalYear_ss,\
                 'signal':signal, 'data_type':sst,'filter':filterName, 'input_signal':input_signal,\
                 'test_split':0, 'iters':1, 'tox_adj':tox_adj_proportion,'longMemory':longMemory,\
                 'n_features':nfeatures, 'FS':feature_selection,'rfe_model':RFE_estimator[0],\
                 'v_start':validationFirstYear_ss, 'v_end':validationFinalYear,'wf_step':wfStep\
                  }
        #cm_y_train = np.array([])
        cm_y_test = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        for j in range(0,len(mmData_v)):
            
            #set last date of training data. -2 because iloc -1 is mmData_v.iloc[j]
            testFinalYear = dataSet.ix[:mmData_v.index[j]].index[-2]
            
            mmData = dataSet.ix[:testFinalYear].reset_index().drop(dropCol, axis=1)
            toxCutoff = getToxCutoff2(abs(dataSet.ix[:testFinalYear].gainAhead),cutoff)
            
            #remove datapoints that dont meet cutoff
            #mmData_adj = mmData[abs(mmData.gainAhead) >=toxCutoff[1]]
            #relabel signals. 
            toxSignals = pd.Series(data=np.where(abs(dataSet.ix[:testFinalYear].gainAhead) >=toxCutoff[1],1,-1), name='signal',index=mmData.index)
                      
            #adjust data proportion
            mmData_adj = adjustDataProportion2(pd.concat([mmData,toxSignals],axis=1), tox_adj_proportion,verbose=False)
            gaSignals = np.where(dataSet.ix[:testFinalYear].reset_index().gainAhead.iloc[mmData_adj.index]>0,1,-1)
            
            dataX = mmData_adj.drop(['signal'], axis=1)
            
                            
            #dy = np.zeros_like(datay_gainAhead.iloc[j])
            #dX = np.zeros_like(dataX)
            X_test = mmData_v.drop(dropCol_v, axis=1).iloc[j].values
            #y_test = np.where(abs(mmData_v.gainAhead.iloc[j])>toxCutoff[1],1,-1)
            y_test = np.where(mmData_v.gainAhead.iloc[j]>0,1,-1)
            X_train = dataX.values
            #y_train = mmData_adj.signal.values
            y_train = gaSignals
            #    runName = 'ShortDataSet_'+'vf_' +vf_name +' df_' + m[0]+'_is'+str(initialInSamplePeriod)+'_oos'+str(wfStep)+sst
            #    model_metrics, sstDictVF[runName] = wf_regress_validate(unfilteredData, sstDF3[sst], [m], model_metrics, initialInSamplePeriod, metaData, showPDFCDF=0)
            if j == 0:
                print "%i starting rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (X_train.shape[0], nrows_oos, wfStep)

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
            
            print j,
            #print 'testFinalYear',testFinalYear,'validationFirstYear', mmData_v.iloc[j].name,
            #print 'Cutoff: ', toxCutoff[1], 'GA', mmData_v.gainAhead.iloc[j],
            #print 'ytrue',y_test,'ypred', y_pred_oos
            
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
        #data is filtered so need to fill in the holes. signal = 0 for days that filtered
        
        st_oos_filt= pd.DataFrame()
        st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
        #st_oos_filt['prior_index'] = pd.Series(cm_y_pred_oos)
        #st_oos_filt['gainAhead'] =  datay_gainAhead[cm_test_index].reset_index().gainAhead
        st_oos_filt.index = mmData_v.index[cm_test_index]

        
        #compute car, show matrix if data is filtered
        print 'Metrics for filtered Validation Datapoints'
        prior_index_filt = pd.concat([st_oos_filt,dataSet.prior_index], axis=1,\
                            join='inner').prior_index.values.astype(int)
        #datay_gainAhead and cm_test_index have the same index. dont need to have same shape because iloc is used in display
        oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                ticker, validationFirstYear_ss, validationFinalYear, iterations, metaData['filter'],show=True)
        CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'LONG', 1)
        CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'SHORT', -1)
        
        #fill in missing data with 0's
        st_oos_filt = pd.concat([st_oos_filt,dataSet.gainAhead,dataSet.prior_index],\
                        axis=1, join='outer').ix[validationFirstYear_ss:validationFinalYear]
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
        #print 'Metrics for', filterName
        #oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1], ticker, validationFirstYear, validationFinalYear, iterations, signal,1)
        #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int), close, 'LONG', 1)
        #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int), close, 'SHORT', -1)
        #update model metrics
        metaData['signal'] = 'LONG 1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead, cm_test_index, m, metaData, CAR25_L1_oos)
        metaData['signal'] = 'SHORT -1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead, cm_test_index, m, metaData, CAR25_Sn1_oos)        
        metaData['signal'] = signal
        #model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead, cm_test_index, m, metaData)
        
        sstDictVF[signal+'_'+filterName+'_' + m[0]+'_'+sst]=st_oos_filt
        
scored_models, bestModel = directional_scoring(model_metrics,filterName)

#save all sst
for m in models:
    if bestModel['params'] == str(m[1]):
        print  '\n\nBest model found...\n', m[1]
        bm = m[1]
print 'Number of features: ', bestModel.n_features, bestModel.FS
print 'WF Signal:', bestModel.signal
print 'WF Initial In-Sample Period:', initialInSamplePeriod
DF3_BMrunName = signal+'_'+filterName+'_'  + bestModel.model + '_'+bestModel.data_type
compareEquity(sstDictVF[DF3_BMrunName],DF3_BMrunName)

sstDictLong = copy.deepcopy(sstDict)
sstDictShort= copy.deepcopy(sstDict)

#save all sst
for runName in sstDict:
    startDate = sstDictVF[DF3_BMrunName].index[0]
    endDate = sstDictVF[DF3_BMrunName].index[-1]
    title = 'L250, S250'
    equityCurves[title] = calcEquity_df(sstDict[runName].ix[startDate:endDate],title)
    
    #long only
    sstDictLong[runName].signals[sstDictLong[runName].signals ==-1]=0
    title = 'L250'
    compareEquity(sstDictLong[runName],title)
    #filename_sst = systemName+ '_' + title + '_' +\
    #       startDate.strftime('%Y%m%d') + 'to' + endDate.strftime('%Y%m%d') + '_' +\
    #       re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDictLong[runName].to_csv(save_path_rs+filename_sst)
    equityCurves[title] = calcEquity_df(sstDictLong[runName].ix[startDate:endDate],title)
    
    #short only
    sstDictShort[runName].signals[sstDictShort[runName].signals ==1]=0
    title = 'S250'
    compareEquity(sstDictShort[runName],title)
    #filename_sst = systemName+ '_' + title + '_' +\
    #   startDate.strftime('%Y%m%d') + 'to' + endDate.strftime('%Y%m%d') + '_' +\
    #   re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDictShort[runName].to_csv(save_path_rs+filename_sst)
    equityCurves[title] = calcEquity_df(sstDictShort[runName].ix[startDate:endDate],title)
    
    #filename_sst = systemName + '_' +title+ '_' +\
    #               validationFirstYear + 'to' + validationFinalYear + '_' +\
    #               re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDict[runName].drop(['prior_index'],axis=1).to_csv(save_path_sst+filename_sst)
    if len(sstDict) >1:
        compareEquity(sstDict[runName],runName)
        
sstDictVF_ = copy.deepcopy(sstDictVF)

for runName in sstDictVF:
    startDate = sstDictVF[runName].index.to_datetime()[0]
    endDate = sstDictVF[runName].index.to_datetime()[-1]
    
    print '\n\n',filterName,' -1 from S250 ONLY'
    compareEquity(sstDictVF[runName],runName)
    filename_sst = ticker + '_' +runName+ '_' +\
                   validationFirstYear_ss + 'to' + validationFinalYear + '_' +\
                   re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDictVF[runName].to_csv(save_path_rs+filename_sst)    

    #all signals
    flatToLongIndex = sstDictVF[runName]\
                    .ix[sstDict[DF1_BMrunName][sstDict[DF1_BMrunName].signals == 1].index].dropna().index
    sstDictVF_[runName].signals.ix[flatToLongIndex] = 1
    title = 'L250, L400(S250), S400(S250)'
    #compareEquity(sstDictVF_[runName],title)
    filename_sst = systemName+ '_' + title + '_' +\
       startDate.strftime('%Y%m%d') + 'to' + endDate.strftime('%Y%m%d') + '_' +\
       re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDictVF_[runName].to_csv(save_path_rs+filename_sst)
    equityCurves[title] = calcEquity_df(sstDictVF_[runName],title)
    
    #flat long
    longToFlatIndex = sstDictVF[runName][sstDictVF[runName].signals ==1].index
    sstDictVF_[runName].signals.ix[longToFlatIndex] = 0
    title = 'L250, FL400(S250), S400(S250)'
    #compareEquity(sstDictVF_[runName],title)
    equityCurves[title] = calcEquity_df(sstDictVF_[runName],title)
    
    #short only
    sstDictVF_[runName].signals[sstDictVF_[runName].signals ==1]=0
    title = 'S400(S250)'
    compareEquity(sstDictVF_[runName],title)
    filename_sst = systemName+ '_' + title + '_' +\
       startDate.strftime('%Y%m%d') + 'to' + endDate.strftime('%Y%m%d') + '_' +\
       re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDictVF_[runName].to_csv(save_path_rs+filename_sst)
    equityCurves[title] = calcEquity_df(sstDictVF_[runName],title)
    
    #long only
    sstDictVF_[runName].signals[sstDictVF_[runName].signals ==0]=1
    sstDictVF_[runName].signals[sstDictVF_[runName].signals ==-1]=0
    
    title = 'L250, L400(S250)'
    compareEquity(sstDictVF_[runName],title)
    filename_sst = systemName+ '_' + title + '_' +\
       startDate.strftime('%Y%m%d') + 'to' + endDate.strftime('%Y%m%d') + '_' +\
       re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    #sstDictVF_[runName].to_csv(save_path_rs+filename_sst)
    equityCurves[title] = calcEquity_df(sstDictVF_[runName],title)
    '''    
    #L120(L250), FS120(L250), L400(S250), S400(S250)
    #get short index from intersect of 400(250) and S120(L250)
    shortToFlatIndex = sstDictVF[runName]\
                    .ix[sstDictDF2_[DF2_BMrunName][sstDictDF2_[DF2_BMrunName].signals == -1].index].dropna().index
    sstDictVF_[runName].signals.ix[shortToFlatIndex] = 0
    flat_shorts120_250 = sstDictVF[runName].copy(deep=True)
    exclude_index = flat_shorts120_250.drop(shortToFlatIndex,axis=0).signals.index
    flat_shorts120_250.signals.ix[exclude_index]=0
    flat_shorts120_250.signals.ix[shortToFlatIndex]=-1
    title = 'S120(L250)'
    compareEquity(flat_shorts120_250,title)    
    title = 'L120(L250), FS120(L250), L400(S250), S400(S250)'
    #compareEquity(sstDictVF_[runName],title)
    equityCurves[title] = calcEquity_df(sstDictVF_[runName],title)
    
    
    #L120(250), L90(120(250)), FS90(120(250)), L400(250), S400(250)
    #reset sstDictVF_
    sstDictVF_ = copy.deepcopy(sstDictVF)
    sstDictVF_[runName].signals.ix[flatToLongIndex] = 1
    shortToFlatIndex = sstDictVF[runName].ix\
                    [sstDictDF4_[DF4_BMrunName][sstDictDF4_[DF4_BMrunName].signals == -1].index].dropna().index
    sstDictVF_[runName].signals.ix[shortToFlatIndex] = 0
    title = 'L120(L250), L90(S120(L250)), FS90(S120(L250)), L400(S250), S400(S250)'
    #compareEquity(sstDictVF_[runName],title)
    equityCurves[title] = calcEquity_df(sstDictVF_[runName],title)
    '''

###############################################
#rank and sort the systems

#make the equity curves the same length
startDate = dataSet.index.to_datetime()[0]
endDate = dataSet.index.to_datetime()[-1]

#set the start/end dates to ones that all equity curves share for apples to apples comparison.
for e in equityCurves:
    if equityCurves[e].index[0]>startDate:
        startDate = equityCurves[e].index[0]
    if equityCurves[e].index[-1]<endDate:
        endDate = equityCurves[e].index[-1]
    #print startDate, equityCurves[e].index[0], equityCurves[e].shape, e
equityCurves_adj = {}
for e in equityCurves:
    equityCurves_adj[e] = equityCurves[e].ix[startDate:]
    #print startDate, equityCurves_adj[e].index[0], equityCurves_adj[e].shape, e

#create equity curve stats    
equityStats = pd.DataFrame(columns=['system','CAR25','CAR50','CAR75','DD100','safef',\
                        'cumCAR','MAXDD','sortinoRatio',\
                       'sharpeRatio','marRatio','k_ratio'], index = range(0,len(equityCurves_adj)))

i=0
for sst in equityCurves_adj:
    startDate = equityCurves_adj[sst].index[0]
    endDate = equityCurves_adj[sst].index[-1]
    years_in_forecast = (endDate-startDate).days/365.0
    #avgSafef = equityCurves_adj[sst].safef.mean()    
    CAR25_metrics = CAR25_df(sst, equityCurves_adj[sst].signals, equityCurves_adj[sst].prior_index.values.astype(int),\
                            close)

    cumCAR = 100*(((equityCurves_adj[sst].equity.iloc[-1]/equityCurves_adj[sst].equity.iloc[0])**(1.0/years_in_forecast))-1.0) 
    MAXDD = max(equityCurves_adj[sst].maxDD)*-100.0
    sortinoRatio = ratio(equityCurves_adj[sst].equity).sortino()
    sharpeRatio = ratio(equityCurves_adj[sst].equity).sharpe()
    marRatio = cumCAR/-MAXDD
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(equityCurves_adj[sst].equity.values)),equityCurves_adj[sst].equity.values)
    k_ratio =(slope/std_err) * math.sqrt(252.0)/len(equityCurves_adj[sst].equity.values)
    
    equityStats.iloc[i].system = sst
    equityStats.iloc[i].CAR25 = CAR25_metrics['CAR25']
    equityStats.iloc[i].CAR50 = CAR25_metrics['CAR50']
    equityStats.iloc[i].CAR75 = CAR25_metrics['CAR75']
    equityStats.iloc[i].DD100 = -CAR25_metrics['DD100']
    equityStats.iloc[i].safef = CAR25_metrics['safef']
    #equityStats.iloc[i].avgSafef = avgSafef
    equityStats.iloc[i].cumCAR = cumCAR
    equityStats.iloc[i].MAXDD = MAXDD
    equityStats.iloc[i].sortinoRatio = sortinoRatio
    equityStats.iloc[i].sharpeRatio = sharpeRatio
    equityStats.iloc[i].marRatio = marRatio
    equityStats.iloc[i].k_ratio = k_ratio
    i+=1

#fill nan to zeros. happens when short system had no short signals
equityStats = equityStats.fillna(0)
#rank the curves based on scoring
#equityStats['avgSafefmm'] =minmax_scale(robust_scale(equityStats.avgSafef.reshape(-1, 1)))
equityStats['CAR25mm'] =minmax_scale(robust_scale(equityStats.CAR25.reshape(-1, 1)))
equityStats['CAR50mm'] =minmax_scale(robust_scale(equityStats.CAR50.reshape(-1, 1)))
equityStats['CAR75mm'] =minmax_scale(robust_scale(equityStats.CAR75.reshape(-1, 1)))
equityStats['DD100mm'] =minmax_scale(robust_scale(equityStats.DD100.reshape(-1, 1)))
equityStats['safefmm'] =minmax_scale(robust_scale(equityStats.safef.reshape(-1, 1)))
equityStats['cumCARmm'] =minmax_scale(robust_scale(equityStats.cumCAR.reshape(-1, 1)))
equityStats['MAXDDmm'] =minmax_scale(robust_scale(equityStats.MAXDD.reshape(-1, 1)))
equityStats['sortinoRatiomm'] = minmax_scale(robust_scale(equityStats.sortinoRatio.reshape(-1, 1)))
equityStats['marRatiomm'] =minmax_scale(robust_scale(equityStats.marRatio.reshape(-1, 1)))
equityStats['sharpeRatiomm'] =minmax_scale(robust_scale(equityStats.sharpeRatio.reshape(-1, 1)))
equityStats['k_ratiomm'] =minmax_scale(robust_scale(equityStats.k_ratio.reshape(-1, 1)))

equityStats['scoremm'] =  equityStats.CAR25mm+\
                        equityStats.DD100mm+equityStats.safefmm+equityStats.cumCARmm+equityStats.MAXDDmm+\
                        equityStats.sortinoRatiomm+equityStats.k_ratiomm+\
                        equityStats.sharpeRatiomm+equityStats.marRatiomm
                        #+equityStats.CAR50mm+equityStats.CAR75mm

                               
equityStats = equityStats.sort_values(['scoremm'], ascending=False)

#################################################
#display charts of ranked systems and benchmarks

#find top system
topSystem = equityStats.system.iloc[0]
#create benchmarks
benchmarks = createBenchmark(dataSet,1.0,'l', startDate,endDate,ticker)
benchmarks[topSystem] = equityCurves_adj[topSystem]
#create yearly stats for benchmark
benchStatsByYear = createYearlyStats(benchmarks)
#create yearly stats for all equity curves with comparison against benchmark
equityCurvesStatsByYear = createYearlyStats(equityCurves_adj, benchStatsByYear)


#display ranked charts
displayRankedCharts(1,benchmarks,benchStatsByYear,equityCurves_adj,equityStats,equityCurvesStatsByYear)
displayRankedCharts(equityStats.system.shape[0],benchmarks,benchStatsByYear,equityCurves_adj,equityStats,equityCurvesStatsByYear)

###################################
#save files

#equity stats
estats_filename = systemName + startDate.strftime("%Y%m%d") +'to'+endDate.strftime("%Y%m%d")+'.csv'
equityStats.to_csv('/media/sf_Python/eStats_'+estats_filename)

# save top systems for DPS/RS
if isinstance(nTopSystems, basestring):
    topSystems = [x for x in equityStats.system]
elif isinstance(nTopSystems, int):
    topSystems = [x for x in equityStats.system.iloc[0:nTopSystems]]

for sst in topSystems:
    
    filename_sst = systemName+ '_' + sst + '_' +\
               startDate.strftime('%Y%m%d') + 'to' + endDate.strftime('%Y%m%d') + '_' +\
               re.sub(r'[^\w]', '', str(dt.today()))[:14] + '.csv'
    equityCurves_adj[sst][['signals','gainAhead']].to_csv(save_path_rs+filename_sst)

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'

