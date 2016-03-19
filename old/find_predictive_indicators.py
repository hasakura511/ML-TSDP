# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:08:04 2015

"""
import math
import random
import numpy as np
import pandas as pd
#import Quandl
import pickle
import re
import talib as ta
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25,\
                            maxCAR25
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score
from sklearn.preprocessing import scale, robust_scale, minmax_scale

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
import pydot

from sknn.mlp import Classifier, Layer
from sknn.backend import lasagne
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFE, RFECV


file_path = '/media/sf_Python/data/from_vf_to_df/'
model_path = '/media/sf_Python/saved_models/DF/'
filteredDataFile = 'LV_F_ES_TOX25_VotingHard_2001-01-11to2015-12-31_20160107004343.csv'
unfilteredDataFile = 'OHLCV_F_ES_TOX25_VotingHard_2001-01-11to2015-12-31_20160107004343.csv'
vf_name = 'VotingHard'
ticker = filteredDataFile[:7]
#vmodel_file = 'VF_20151215193820622423GBCmodelGBCsignalTOX250011497tox_adj1date_end20150331tickerF_SPdate__start20090101Name27dtypeobject.pkl'
signal = 'beLong'
DD95_limit = 0.20
initial_equity = 100000.0
#dataLoadStartDate = "1998-12-22"
#dataLoadEndDate = "2016-01-04"

#offline learning
testFirstYear = "2014-01-01"
testFinalYear = "2014-12-31"
test_split = 0 #test split
iterations = 1 #for sss
tox_adj_proportion = 0
start_nfeatures = 41 # start loop
nfeatures = 41 # end loop
ShowInSample = False
#online learning
validationFirstYear ="2015-01-01"
validationFinalYear ="2015-06-30"

RSILookback = 1.5
zScoreLookback = 20
ATRLookback = 5
beLongThreshold = 0.0
DPOLookback = 3
# relATR -> 10 zscore atr lookback + shift 5 + 10zscore atr rel lookback = 25
#dc phase =82
maxlb = max(RSILookback, zScoreLookback, ATRLookback, DPOLookback, 82)

start_time = time.time()
model_metrics = init_report()
RFE_estimator = LinearRegression()
rfe_models = [ #commented ones gives error sometimes. 
        ##("RandomForestRegressor",RandomForestRegressor()),\
        #("AdaBoostRegressor",AdaBoostRegressor()),\ #Input contains NaN, infinity or a value too large for dtype('float64').
        #("ExtraTreesRegressor",ExtraTreesRegressor()),\
        ("GradientBoostingRegressor",GradientBoostingRegressor()),\
        ##("IsotonicRegression",IsotonicRegression()), #ValueError: X should be a 1d array\
        ("DecisionTreeRegressor",DecisionTreeRegressor()),\
        ("ExtraTreeRegressor",ExtraTreeRegressor()),\
        #("Ridge",Ridge()),\
        #("LinearRegression",LinearRegression()),\
        ##("Lasso",Lasso()),\ #works but looks like binary class
        ##("LassoLars",LassoLars()),\ ##works looks like binary class
        ("BayesianRidge", BayesianRidge()),\
        #("PassiveAggressiveRegressor",PassiveAggressiveRegressor()),\
        #("SGDRegressor",SGDRegressor()),\
        #("TheilSenRegressor",TheilSenRegressor()),\
        #("LinearSVR", LinearSVR()),\
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
        ("RandomForestRegressor",RandomForestRegressor()),\
        ("AdaBoostRegressor",AdaBoostRegressor()),\
        ("BaggingRegressor",BaggingRegressor()),\
        ("ExtraTreesRegressor",ExtraTreesRegressor()),\
        ("GradientBoostingRegressor",GradientBoostingRegressor()),\
        #("IsotonicRegression",IsotonicRegression()),\ #ValueError: X should be a 1d array
        #("KernelRidge",KernelRidge()),\
        ("DecisionTreeRegressor",DecisionTreeRegressor()),\
        #("ExtraTreeRegressor",ExtraTreeRegressor()),\
        #("ARDRegression", ARDRegression()),\
        #("LogisticRegression", LogisticRegression()),\ # ValueError("Unknown label type: %r" % y)
        ("Ridge",Ridge()),\
        #("RANSACRegressor",RANSACRegressor()),\
        #("LinearRegression",LinearRegression()),\
        #("Lasso",Lasso()),\
        #("LassoLars",LassoLars()),\
        #("BayesianRidge", BayesianRidge()),\
        #("PassiveAggressiveRegressor",PassiveAggressiveRegressor()),\ #all zero array
        #("SGDRegressor",SGDRegressor()),\
        #("TheilSenRegressor",TheilSenRegressor()),\
        ("KNeighborsRegressor", KNeighborsRegressor()),\
        #("RadiusNeighborsRegressor",RadiusNeighborsRegressor()),\
        ("LinearSVR", LinearSVR()),\
        #("rbf1SVR",SVR(kernel='rbf', C=1, gamma=0.1)),\ #too slow
        #("rbf10SVR",SVR(kernel='rbf', C=10, gamma=0.1)),\#too slow
        #("rbf100SVR",SVR(kernel='rbf', C=1e3, gamma=0.1)),\#too slow
        #("polySVR",SVR(kernel='poly', C=1e3, degree=2)),\#too slow
         ]

vfData = pd.read_csv(file_path+filteredDataFile, index_col='dates')
vfData['LongGT0'] = np.where(vfData.gainAhead>0,1,-1)

qt = pd.read_csv(file_path+unfilteredDataFile, index_col='dates')
#transform functions workonly with numerical indexes starting from zero

unfilteredData = pd.concat([qt[' OPEN'],qt[' HIGH'],qt[' LOW'],qt[' CLOSE'],qt[' VOL']], axis=1)
unfilteredData.columns = ['Open','Low','High','Close','Volume']

Open = unfilteredData.Open.values
low = unfilteredData.Low.values
high = unfilteredData.High.values
close = unfilteredData.Close.values
volume = unfilteredData.Volume.values

unfilteredData['pctChangeOpen'] = priceChange(Open)
unfilteredData['pctChangeLow'] = priceChange(low)
unfilteredData['pctChangeHigh'] = priceChange(high)
#transform relative to close
#PRICE LEVEL
unfilteredData['linreg'] = zScore(ta.LINEARREG(close, timeperiod=7)/close,zScoreLookback)
unfilteredData['tsf'] = zScore(ta.TSF(close, timeperiod=7)/close,zScoreLookback)
unfilteredData['support'], unfilteredData['resistance']= ta.MINMAX(close, timeperiod=10)/close
unfilteredData['support']=zScore(unfilteredData['support'],zScoreLookback)
unfilteredData['resistance']=zScore(unfilteredData['resistance'],zScoreLookback)
unfilteredData['bbupper'], unfilteredData['bbmiddle'], unfilteredData['bblower'] = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)/close
unfilteredData['bbupper']=zScore(unfilteredData['bbupper'],zScoreLookback)
unfilteredData['bbmiddle']=zScore(unfilteredData['bbmiddle'],zScoreLookback)
unfilteredData['bblower']=zScore(unfilteredData['bblower'],zScoreLookback)

# transform to zs
#MOMENTUM
unfilteredData['roc'] = zScore(ta.ROC(close, timeperiod=2),20)
unfilteredData['rocp'] = zScore(ta.ROCP(close, timeperiod=3),20)
unfilteredData['rocr'] = zScore(ta.ROCR(close, timeperiod=4),20)
unfilteredData['rocr100'] = zScore(ta.ROCR100(close, timeperiod=5),20)
unfilteredData['trix'] = zScore(ta.TRIX(close, timeperiod=2),20)
unfilteredData['ultosc'] = zScore(ta.ULTOSC(high, low, close, timeperiod1=3, timeperiod2=5, timeperiod3=10),20)
unfilteredData['willr'] = zScore(ta.WILLR(high, low, close, timeperiod=7),20)
unfilteredData['linreg_slope'] = zScore(ta.LINEARREG_SLOPE(close, timeperiod=7),20)
unfilteredData['adx'] = zScore(ta.ADX(high, low, close, timeperiod=5),20)
unfilteredData['adxr'] = zScore(ta.ADXR(high, low, close, timeperiod=3),20)
unfilteredData['macd'], unfilteredData['macdsignal'], unfilteredData['macdhist'] = ta.MACD(close, fastperiod=3, slowperiod=7, signalperiod=3)
unfilteredData['macd'] = zScore(unfilteredData['macd'],20)
unfilteredData['macdsignal'] = zScore(unfilteredData['macdsignal'],20)
unfilteredData['macdhist'] = zScore(unfilteredData['macdhist'],20)
unfilteredData['minus_di'] = zScore(ta.MINUS_DI(high, low, close, timeperiod=7),20)
unfilteredData['minus_dm'] = zScore(ta.MINUS_DM(high, low, timeperiod=7),20)
unfilteredData['cmo'] = zScore(ta.CMO(close, timeperiod=3),20)
unfilteredData['mom'] = zScore(ta.MOM(close, timeperiod=3),20)
unfilteredData['plus_di'] = zScore(ta.PLUS_DI(high, low, close, timeperiod=5),20)
unfilteredData['plus_dm'] = zScore(ta.PLUS_DM(high, low, timeperiod=5),20)
unfilteredData['ppo'] = zScore(ta.PPO(close, fastperiod=5, slowperiod=10, matype=0),20)
#Volume
#unfilteredData['ad'] = zScore(ta.AD(high, low, close, volume),20)
#unfilteredData['adosc'] = zScore(ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10),20)
unfilteredData['obv'] = zScore(ta.OBV(close, volume),20)
#VOLATILITY
unfilteredData['atr'] = zScore(ta.ATR(high, low, close, timeperiod=7),20)
unfilteredData['natr'] = zScore(ta.NATR(high, low, close, timeperiod=10),20)
unfilteredData['trange'] = zScore(ta.TRANGE(high, low, close),20)
#CYCLE
unfilteredData['ht_dcperiod'] = zScore(ta.HT_DCPERIOD(close),20)
unfilteredData['ht_dcphase'] = zScore(ta.HT_DCPHASE(close),20)
unfilteredData['inphase'], unfilteredData['quadrature'] = ta.HT_PHASOR(close)
unfilteredData['inphase']=zScore(unfilteredData['inphase'],20)
unfilteredData['quadrature']=zScore(unfilteredData['quadrature'],20)

# div 100
#MOMENTUM
unfilteredData['apo'] = ta.APO(close, fastperiod=3, slowperiod=10, matype=0)/100.0
unfilteredData['aroondown'], unfilteredData['aroonup'] = ta.AROON(high, low, timeperiod=7)
unfilteredData['aroondown']=unfilteredData['aroondown']/100.0
unfilteredData['aroonup']=unfilteredData['aroonup']/100.0
unfilteredData['aroonosc'] = ta.AROONOSC(high, low, timeperiod=5)/100.0
unfilteredData['cci'] = ta.CCI(high, low, close, timeperiod=5)/100.0
unfilteredData['dx'] = ta.DX(high, low, close, timeperiod=5)/100.0
unfilteredData['mfi'] = ta.MFI(high, low, close, volume, timeperiod=7)/100.0
unfilteredData['rsi'] = ta.RSI(close, timeperiod=14)/100.0
unfilteredData['slowk'], unfilteredData['slowd'] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
unfilteredData['slowk']=unfilteredData['slowk']/100.0
unfilteredData['slowd']=unfilteredData['slowd']/100.0
unfilteredData['fastk'], unfilteredData['fastd'] = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
unfilteredData['fastk']=unfilteredData['fastk']/100.0
unfilteredData['fastd']=unfilteredData['fastd']/100.0
unfilteredData['rsifastk'], unfilteredData['rsifastd'] = ta.STOCHRSI(close, timeperiod=10, fastk_period=5, fastd_period=3, fastd_matype=0)
unfilteredData['rsifastk']=unfilteredData['rsifastk']/100.0
unfilteredData['rsifastd']=unfilteredData['rsifastd']/100.0

#no transform
unfilteredData['bop'] = ta.BOP(Open, high, low, close)
#CYCLE
unfilteredData['sine'], unfilteredData['leadsine'] = ta.HT_SINE(close)
unfilteredData['ht_trendmode'] = ta.HT_TRENDMODE(close)

       
#unfilteredData.to_csv('/media/sf_Python/indicators.csv')
# sanity check the file
#pd.concat([filteredData,unfilteredData], axis=1, join='outer').to_csv('/media/sf_Python/indexcheck_outer_join.csv')
#pd.concat([filteredData,unfilteredData], axis=1, join='inner').to_csv('/media/sf_Python/indexcheck_inner_join.csv')

for col in unfilteredData:
    if sum(np.isnan(unfilteredData[col][maxlb:].values))>0:
        print unfilteredData[col][maxlb:][np.isnan(unfilteredData[col][maxlb:].values)]
        raise ValueError, 'nan in %s' % col
    elif sum(np.isinf(unfilteredData[col][maxlb:].values))>0:
        print unfilteredData[col][maxlb:][np.isnan(unfilteredData[col][maxlb:].values)]
        raise ValueError, 'inf in %s' % col
    elif sum(np.isneginf(unfilteredData[col][maxlb:].values))>0:
        print unfilteredData[col][maxlb:][np.isnan(unfilteredData[col][maxlb:].values)]
        raise ValueError, '-inf in %s' % col
        
#add new features
filteredData = pd.concat([vfData,unfilteredData], axis=1, join='inner').dropna()

close = unfilteredData.reset_index().Close
dataSet = pd.concat([filteredData[' OPEN'],filteredData[' HIGH'],filteredData[' LOW'],filteredData[' CLOSE'],filteredData[' VOL'], filteredData['Unnamed: 0']], axis=1)
dataSet.columns = ['Open','High','Low','Close','Volume','prior_index']

nrows = dataSet.shape[0]
print "Successfully loaded %i rows" % nrows

#no shifting allowed here because it is filtered and data has gaps
#dataSet.Close = filteredData.Close
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

#transform relative to close
#PRICE LEVEL
dataSet['linreg']=filteredData['linreg']
dataSet['tsf']=filteredData['tsf']
dataSet['support'] =filteredData['support']
dataSet['resistance']=filteredData['resistance']
dataSet['bbupper']=filteredData['bbupper']
dataSet['bbmiddle']=filteredData['bbmiddle']
dataSet['bblower']=filteredData['bblower']

#transform to minmax
#MOMENTUM
dataSet['roc']=filteredData['roc']
dataSet['rocp']=filteredData['rocp']
dataSet['rocr']=filteredData['rocr']
dataSet['rocr100']=filteredData['rocr100']
dataSet['trix'] =filteredData['trix']
dataSet['ultosc'] =filteredData['ultosc']
dataSet['willr']=filteredData['willr']

# transform to zs
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

#dataSet['signal'] = filteredData[signal]

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

mData = dataSet.drop(['Open','High','Low','Close',
                       'Volume'],
                        axis=1) 

#  Select the date range to test
mmData_t = mData.ix[testFirstYear:testFinalYear].dropna()
mmData_v = mData.ix[validationFirstYear:validationFinalYear].dropna()

datay_t = mmData_t.gainAhead
mmData_t = mmData_t.drop(['prior_index','gainAhead'],axis=1)
dataX_t = mmData_t

nrows_t = mmData_t.shape[0]
print " Training Set %i rows.." % nrows_t
cols = dataX_t.columns.shape[0]
feature_names = []
print '\nTraining using %i features: ' % cols
for i,x in enumerate(dataX_t.columns):
    print i+1,x+',',
    feature_names = feature_names+[x]

#  Copy from pandas dataframe to numpy arrays
dy_t = np.zeros_like(datay_t)
dX_t = np.zeros_like(dataX_t)

dy_t = datay_t.values
dX_t = dataX_t.values

datay_v = mmData_v.gainAhead
mmData_v = mmData_v.drop(['prior_index','gainAhead'],axis=1)
dataX_v = mmData_v

nrows_v = mmData_v.shape[0]
print " Validation Set %i rows.." % nrows_v

#  Copy from pandas dataframe to numpy arrays
dy_v = np.zeros_like(datay_v)
dX_v = np.zeros_like(dataX_v)

dy_v = datay_v.values
dX_v = dataX_v.values

'''
for nfeatures in range(start_nfeatures,nfeatures+1):
    #  by univariate
    for m in models:
        print m[0]
        y_test_true_class = np.array(([-1 if x<0 else 1 for x in dy_t]))
        
        #Univariate feature selection
        dX_all = np.vstack((dX_t, dX_v))
        dy_all = np.hstack((dy_t, dy_v))
        skb = SelectKBest(f_regression, k=nfeatures)
        skb.fit(dX_t, dy_t)
        X_new = skb.transform(dX_all)
        featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
        print 'Top %i univariate features' % len(featureRank)
        print featureRank 
        dX_t_rfe = X_new[range(0,dX_t.shape[0])]
        dX_v_rfe = X_new[dX_t.shape[0]:]
    
        m[1].fit(dX_t_rfe, dy_t)
        
        if m[0][:2] == 'GA':
            print '\nProgram:', m[1]._program
            
        plot_learning_curve(m[1], m[0], X_new,dy_all, scoring='r2')
        
        try:
            print 'Train R^2:    ', m[1].score(dX_t_rfe,dy_t)
            print 'Validation R^2:    ', m[1].score(dX_v_rfe,dy_v)
        except Exception, err:
            print err
            pass
            #graph = pydot.graph_from_dot_data(m[1]._program.export_graphviz())
            #display_png(graph.create_png(), raw=True) 
    
        #  test the in-sample fit
        y_pred_is = m[1].predict(dX_t_rfe)
        y_pred_class_is = np.array(([-1 if x<0 else 1 for x in y_pred_is]))
        y_true_class_is = np.array(([-1 if x<0 else 1 for x in dy_t]))
    
        y_pred_oos = m[1].predict(dX_v_rfe)
        y_pred_class_oos = np.array(([-1 if x<0 else 1 for x in y_pred_oos]))
        y_true_class_oos = np.array(([-1 if x<0 else 1 for x in dy_v]))
        
        #cm_is = confusion_matrix(y_train, y_pred_is)
        #cm_sum_is = cm_sum_is + cm_is
        if ShowInSample:
            metaData = {'ticker':ticker, 'date__start':testFirstYear, 'date_end':testFinalYear,\
             'signal':signal, 'rows':nrows_t,'total_cols':cols, \
             'n_features':nfeatures, 'FS':'Univariate', 'featureRank':str(featureRank)}
            #plot in-sample data    
            plt.figure()
            coef, b = np.polyfit(y_pred_is, dy_t, 1)
            plt.title('In-Sample')
            plt.ylabel('gainAhead')
            plt.xlabel('ypred gainAhead')
            plt.plot(y_pred_is, dy_t, '.')
            plt.plot(y_pred_is, coef*y_pred_is + b, '-')
            plt.show()
            print 'Training Error: ', sum((y_pred_is -  dy_t)**2)/y_pred_is.shape[0]
            is_display_cmatrix2(y_true_class_is, y_pred_class_is, datay_t.reset_index().gainAhead, datay_t.reset_index().gainAhead.index, m[0], ticker, testFirstYear, testFinalYear, iterations, signal)
            model_metrics = update_report(model_metrics, "IS", y_pred_class_is, y_true_class_is, datay_t.reset_index().gainAhead, datay_t.reset_index().gainAhead.index, m, metaData)
    
        metaData = {'ticker':ticker, 'date__start':validationFirstYear, 'date_end':validationFinalYear,\
             'signal':signal, 'rows':nrows_v,'total_cols':cols, \
             'n_features':nfeatures, 'FS':'Univariate', 'featureRank':str(featureRank)}
        #plot out-of-sample data
        plt.figure()
        coef, b = np.polyfit(y_pred_oos, dy_v, 1)
        plt.title('Out-of-Sample')
        plt.ylabel('gainAhead')
        plt.xlabel('ypred gainAhead')
        plt.plot(y_pred_oos, dy_v, '.')
        plt.plot(y_pred_oos, coef*y_pred_oos + b, '-')
        plt.show()
        print 'Validation Error: ', sum((y_pred_oos -  dy_v)**2)/y_pred_oos.shape[0]
        oos_display_cmatrix2(y_true_class_oos, y_pred_class_oos, datay_v.reset_index().gainAhead, datay_v.reset_index().gainAhead.index, m[0], ticker, validationFirstYear, validationFinalYear, iterations, signal)
        model_metrics = update_report(model_metrics, "OOS", y_pred_class_oos, y_true_class_oos, datay_v.reset_index().gainAhead, datay_v.reset_index().gainAhead.index, m, metaData)
'''
#  by RFECV
for rfm in rfe_models:
    print 'using', rfm[0], 'for RFECV feature selection'
    RFE_estimator = rfm[1]
    for m in models:
        print 'Training using', m[0]
        y_test_true_class = np.array(([-1 if x<0 else 1 for x in dy_t]))
        
        #Recursive feature elimination with cross-validation: 
        #A recursive feature elimination example with automatic tuning of the
        #number of features selected with cross-validation.
        rfe = RFECV(estimator=RFE_estimator, step=1)
        rfe.fit(dX_t, y_test_true_class)
        #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
        featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
        print 'Top %i RFECV features' % len(featureRank)
        print featureRank    
     
        dX_all = np.vstack((dX_t, dX_v))
        dy_all = np.hstack((dy_t, dy_v))
        X_new = rfe.transform(dX_all)
 
        dX_t_rfe = X_new[range(0,dX_t.shape[0])]
        dX_v_rfe = X_new[dX_t.shape[0]:]
    
        m[1].fit(dX_t_rfe, dy_t)
        
        if m[0][:2] == 'GA':
            print '\nProgram:', m[1]._program
            
        plot_learning_curve(m[1], m[0], X_new,dy_all, scoring='r2')
        
        try:
            print 'Train R^2:    ', m[1].score(dX_t_rfe,dy_t)
            print 'Validation R^2:    ', m[1].score(dX_v_rfe,dy_v)
        except Exception, err:
            print err
            pass 

        #  test the in-sample fit
        y_pred_is = m[1].predict(dX_t_rfe)
        y_pred_class_is = np.array(([-1 if x<0 else 1 for x in y_pred_is]))
        y_true_class_is = np.array(([-1 if x<0 else 1 for x in dy_t]))

        y_pred_oos = m[1].predict(dX_v_rfe)
        y_pred_class_oos = np.array(([-1 if x<0 else 1 for x in y_pred_oos]))
        y_true_class_oos = np.array(([-1 if x<0 else 1 for x in dy_v]))
        
        #cm_is = confusion_matrix(y_train, y_pred_is)
        #cm_sum_is = cm_sum_is + cm_is
        if ShowInSample:
            metaData = {'ticker':ticker, 'date__start':testFirstYear, 'date_end':testFinalYear,\
             'signal':signal, 'rows':nrows_t,'total_cols':cols, 'rfe_model':rfm[0],\
             'n_features':len(featureRank), 'FS':'RFECV', 'featureRank':str(featureRank)}
            #plot in-sample data    
            plt.figure()
            coef, b = np.polyfit(y_pred_is, dy_t, 1)
            plt.title('In-Sample')
            plt.ylabel('gainAhead')
            plt.xlabel('ypred gainAhead')
            plt.plot(y_pred_is, dy_t, '.')
            plt.plot(y_pred_is, coef*y_pred_is + b, '-')
            plt.show()
            print 'Training Error: ', sum((y_pred_is -  dy_t)**2)/y_pred_is.shape[0]
            is_display_cmatrix2(y_true_class_is, y_pred_class_is, datay_t.reset_index().gainAhead, datay_t.reset_index().gainAhead.index, m[0], ticker, testFirstYear, testFinalYear, iterations, signal)
            model_metrics = update_report(model_metrics, "IS", y_pred_class_is, y_true_class_is, datay_t.reset_index().gainAhead, datay_t.reset_index().gainAhead.index, m, metaData)

        metaData = {'ticker':ticker, 'date__start':validationFirstYear, 'date_end':validationFinalYear,\
             'signal':signal, 'rows':nrows_v,'total_cols':cols, 'rfe_model':rfm[0],\
             'n_features':len(featureRank), 'FS':'RFECV','featureRank':str(featureRank)}
        #plot out-of-sample data
        plt.figure()
        coef, b = np.polyfit(y_pred_oos, dy_v, 1)
        plt.title('Out-of-Sample')
        plt.ylabel('gainAhead')
        plt.xlabel('ypred gainAhead')
        plt.plot(y_pred_oos, dy_v, '.')
        plt.plot(y_pred_oos, coef*y_pred_oos + b, '-')
        plt.show()
        print 'Validation Error: ', sum((y_pred_oos -  dy_v)**2)/y_pred_oos.shape[0]
        oos_display_cmatrix2(y_true_class_oos, y_pred_class_oos, datay_v.reset_index().gainAhead, datay_v.reset_index().gainAhead.index, m[0], ticker, validationFirstYear, validationFinalYear, iterations, signal)
        model_metrics = update_report(model_metrics, "OOS", y_pred_class_oos, y_true_class_oos, datay_v.reset_index().gainAhead, datay_v.reset_index().gainAhead.index, m, metaData)

'''
#  by RFE
for m in models:
    print m[0]
    y_test_true_class = np.array(([-1 if x<0 else 1 for x in dy_t]))
    
    # Manually set the Recursive feature elimination
    rfe = RFE(estimator=RFE_estimator, n_features_to_select=nfeatures, step=1)
    rfe.fit(dX_t, y_test_true_class)
    #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
    featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
    print 'Top %i features' % len(featureRank)
    print featureRank    
    
    dX_all = np.vstack((dX_t, dX_v))
    dy_all = np.hstack((dy_t, dy_v))
    X_new = rfe.transform(dX_all)

    dX_t_rfe = X_new[range(0,dX_t.shape[0])]
    dX_v_rfe = X_new[dX_t.shape[0]:]

    m[1].fit(dX_t_rfe, dy_t)
    
    if m[0][:2] == 'GA':
        print '\nProgram:', m[1]._program
        
    plot_learning_curve(m[1], m[0], X_new,dy_all, scoring='r2')
    
    try:
        print 'Train R^2:    ', m[1].score(dX_t_rfe,dy_t)
        print 'Validation R^2:    ', m[1].score(dX_v_rfe,dy_v)
    except Exception, err:
        print err
        pass 

    #  test the in-sample fit
    y_pred_is = m[1].predict(dX_t_rfe)
    y_pred_class_is = np.array(([-1 if x<0 else 1 for x in y_pred_is]))
    y_true_class_is = np.array(([-1 if x<0 else 1 for x in dy_t]))

    y_pred_oos = m[1].predict(dX_v_rfe)
    y_pred_class_oos = np.array(([-1 if x<0 else 1 for x in y_pred_oos]))
    y_true_class_oos = np.array(([-1 if x<0 else 1 for x in dy_v]))
    
    #cm_is = confusion_matrix(y_train, y_pred_is)
    #cm_sum_is = cm_sum_is + cm_is
    if ShowInSample:
        metaData = {'ticker':ticker, 'date__start':testFirstYear, 'date_end':testFinalYear,\
         'signal':signal, 'rows':nrows_t,'total_cols':cols, \
         'n_features':nfeatures, 'FS':'RFE','featureRank':str(featureRank)}
        #plot in-sample data    
        plt.figure()
        coef, b = np.polyfit(y_pred_is, dy_t, 1)
        plt.title('In-Sample')
        plt.ylabel('gainAhead')
        plt.xlabel('ypred gainAhead')
        plt.plot(y_pred_is, dy_t, '.')
        plt.plot(y_pred_is, coef*y_pred_is + b, '-')
        plt.show()
        print 'Training Error: ', sum((y_pred_is -  dy_t)**2)/y_pred_is.shape[0]
        is_display_cmatrix2(y_true_class_is, y_pred_class_is, datay_t.reset_index().gainAhead, datay_t.reset_index().gainAhead.index, m[0], ticker, testFirstYear, testFinalYear, iterations, signal)
        model_metrics = update_report(model_metrics, "IS", y_pred_class_is, y_true_class_is, datay_t.reset_index().gainAhead, datay_t.reset_index().gainAhead.index, m, metaData)

    metaData = {'ticker':ticker, 'date__start':validationFirstYear, 'date_end':validationFinalYear,\
         'signal':signal, 'rows':nrows_v,'total_cols':cols, \
         'n_features':nfeatures, 'FS':'RFE','featureRank':str(featureRank)}
    #plot out-of-sample data
    plt.figure()
    coef, b = np.polyfit(y_pred_oos, dy_v, 1)
    plt.title('Out-of-Sample')
    plt.ylabel('gainAhead')
    plt.xlabel('ypred gainAhead')
    plt.plot(y_pred_oos, dy_v, '.')
    plt.plot(y_pred_oos, coef*y_pred_oos + b, '-')
    plt.show()
    print 'Validation Error: ', sum((y_pred_oos -  dy_v)**2)/y_pred_oos.shape[0]
    oos_display_cmatrix2(y_true_class_oos, y_pred_class_oos, datay_v.reset_index().gainAhead, datay_v.reset_index().gainAhead.index, m[0], ticker, validationFirstYear, validationFinalYear, iterations, signal)
    model_metrics = update_report(model_metrics, "OOS", y_pred_class_oos, y_true_class_oos, datay_v.reset_index().gainAhead, datay_v.reset_index().gainAhead.index, m, metaData)
'''
filename_fpi = ticker + '_' +vf_name +'_' + '_' +\
                   validationFirstYear + 'to' + validationFinalYear + '_' +\
                   re.sub(r'[^\w]', '', str(datetime.datetime.today()))[:14] + '.csv'
                   
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
model_metrics.to_csv('/media/sf_Python/FPI_'+filename_fpi)