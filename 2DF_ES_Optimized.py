# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

import numpy as np
import math
import talib as ta
import pandas as pd
from suztoolz.transform import zigzag as zg
import arch
from os import listdir
from os.path import isfile, join

from datetime import datetime
import matplotlib.pyplot as plt
#from pandas.io.data import DataReader
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import time
from suztoolz.transform import adf_test
from suztoolz.loops import CAR25, CAR25_prospector, maxCAR25
from suztoolz.display import init_report, update_report_prospector,\
                            display_CAR25
from sklearn.grid_search import ParameterGrid
import re

#read data start date
start_date = 19981222

#main ticker
filename = 'F_ES.txt'
ticker = filename[:-4]

#append price changes from files from this folder
from_quantiacs = 'D:/DropBox/sharedTSDP/data/tickerData/'
from_quandl = 'D:/Dropbox/SharedTSDP/data/quandl/'
from_fileprep = 'D:/Dropbox/SharedTSDP/data/from_fileprep/'

#add aux data
add_aux_quantiacs = ['F_ED','F_GC','F_DX','F_CL','F_US']
add_aux_quandl = ['HSI', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
#add aux series if difference is less than datapoint_threshold
#datapoint_threshold = -100 
#candle_threshold = .1 #keep candle data if > 10% series


#zigzag steps to check for non-vf, directional trends
#zz_steps = [0.005]

#find signals with the highest CAR25
#direction = 'LONG'

#years_in_study = int(str(end_date)[:4]) - int(str(start_date)[:4])

hold_days = 1
system_accuracy = .55
DD95_limit = 0.20
initial_equity = 1.0
fraction = 1.00
forecast_horizon = 50 #  trading days
number_forecasts = 20  # Number of simulated forecasts

start_time = time.time()
print 'Load primary data file... %s' % from_quantiacs+filename
series = pd.read_csv(from_quantiacs+filename, error_bad_lines=False, index_col='DATE')
end_date = series.index[-1]

series[ticker] = series[' CLOSE'].pct_change()
for col in series:
    if col == ' P' or col == ' R' or col == ' RINFO':
        #print 'deleting columns', col
        series = series.drop(col,1)
f_count = series[ticker].count()

#load data from quantiacs
start_date_aux = 0
aux_files = [ f for f in listdir(from_quantiacs) if isfile(join(from_quantiacs,f)) ]

for i,f in enumerate(aux_files):
#    if f != filename:
    if f[:-4] in add_aux_quantiacs:
        #print f
        add_series = pd.read_csv(from_quantiacs+f, error_bad_lines=False, index_col='DATE')
        add_series[f[:-4]] = add_series[' CLOSE']
        for col in add_series:         
            if col != f[:-4]:
                #print 'del', col
                del add_series[col]
        if (add_series.index[0]<=start_date and add_series.index[-1]>=end_date):
            if add_series.index[0]>start_date_aux:
                start_date_aux = add_series.index[0]
            print 'adding ', f[:-4], add_series.count()[0],'rows to ',ticker, f_count, ' rows..'  
            series = pd.merge(series, add_series, left_index=True, right_index=True, how='outer')
        else:
            if add_series.index[0]>start_date:            
                print 'skipping ', f[:-4], ' as ' , add_series.index[0], ' > ', start_date
            if add_series.index[-1]<end_date:
                print 'skipping ', f[:-4], ' as ' , add_series.index[-1], ' < ', end_date

#load data from quandl
aux_files = [ f for f in listdir(from_quandl) if isfile(join(from_quandl,f)) ]
#merge everything
#add_aux = [f[:-4] for f in aux_files]

for i,f in enumerate(aux_files):
#    if f != filename:
    if f[:-4] in add_aux_quandl:
        #print f
        add_series = pd.read_csv(from_quandl+f, error_bad_lines=False, index_col='Date')
        add_series.index = add_series.index.map(lambda x: x.replace('-','')).astype(int)
        add_series['S_'+f[:-4]] = add_series['Close']
        for col in add_series:         
            if col != 'S_'+f[:-4]:
                #print 'del', col
                del add_series[col]
        if (add_series.index[0]<=start_date and add_series.index[-1]>=end_date):
            if add_series.index[0]>start_date_aux:
                start_date_aux = add_series.index[0]
            print 'adding ', f[:-4], add_series.count()[0],'rows to ',ticker, f_count, ' rows..'  
            series = pd.merge(series, add_series, left_index=True, right_index=True, how='outer')
        else:
            if add_series.index[0]>start_date:            
                print 'skipping ', f[:-4], ' as ' , add_series.index[0], ' > ', start_date
            if add_series.index[-1]<end_date:
                print 'skipping ', f[:-4], ' as ' , add_series.index[-1], ' < ', end_date
            
open= series[' OPEN'].values
high =series[' HIGH'].values
low=series[' LOW'].values
close=series[' CLOSE'].values
volume=np.array([float(x) for x in series[' VOL'].values])
oi=series[' OI'].values

#cut off nans from in the beginning of the dataset
series = series.ix[start_date_aux:]
#drop nans from data not in ticker
drop_index = series.index[np.isnan(series[ticker].values)]
series = series.drop(drop_index, axis=0)

for col in series.columns:
    for i in series.reset_index().index[np.isnan(series[col].values)]:
        series[col].iloc[i] = series[col].iloc[i-1]
'''
#zz_returns = pd.DataFrame()
zz_signals = pd.DataFrame()
#zz_price_chg = pd.DataFrame()
#pivot_df = pd.DataFrame()
print 'Creating Signal labels..',
for i in zz_steps:
    for j in zz_steps:
        label = 'ZZ '+str(i) + ',-' + str(j)
        print label,
        #pivot_df[label] = zg(close, i, -j).peak_valley_pivots()
        #print label, zg(close, i, -j).compute_segment_returns().shape
        #g = pd.DataFrame(zg(close, i, -j).compute_segment_returns(), columns=[pivots])
        #zz_returns = pd.concat([zz_returns,g],ignore_index=True, axis=1)
        #get signals and delete first row to align signalAhead
        zz_signals[label] = np.delete(zg(close, i, -j).pivots_to_modes(),0,0)

#filename = ticker+'_sigcheck.csv'
sig_check = pd.concat([series,zz_signals], axis=1)
#sig_check.dropna().to_csv(filename, index=False)
#print 'Successfully created ', filename
'''
#data_cons = pd.concat([y,series,data,data2,data3,data4], axis=1)

#data_cons.dropna().to_csv(filename[:-4]+'_.csv')

#filename = ticker+'_sigcheck.csv'

print '\n\nNew simulation run '
#print 'Testing profit potential for %s positions...\n ' % direction
print 'Issue:             ' + ticker
print 'Dates:             ' + str(start_date)
print '  to:              ' + str(end_date)
print 'Hold Days:          %i ' % hold_days
print 'System Accuracy:    %.2f ' % system_accuracy
print 'DD 95 limit:        %.2f ' % DD95_limit
print 'Forecast Horizon:   %i ' % forecast_horizon
print 'Number Forecasts:   %i ' % number_forecasts
print 'Initial Equity:     %i ' % initial_equity

# ------------------------
#  Variables used for simulation

qt = series.dropna()
mask = (qt.index > start_date) & (qt.index <= end_date)
qt = qt.loc[mask]
#print qt.shape
#print qt.head()


#print(qt.loc[mask])

qtC = qt.reset_index()[' CLOSE']
qtP = qt.reset_index()[ticker]
nrows = qtC.shape[0]
print 'Number Rows:        %d ' % nrows

number_trades = forecast_horizon / hold_days
number_days = number_trades*hold_days
print 'Number Days:        %i ' % number_days
print 'Number Trades:      %d ' % number_trades

al = number_days+1
#   These arrays are the number of days in the forecast
account_balance = np.zeros(al, dtype=float)     # account balance

pltx = np.zeros(al, dtype=float)
plty = np.zeros(al, dtype=float)

max_IT_DD = np.zeros(al, dtype=float)     # Maximum Intra-Trade drawdown
max_IT_Eq = np.zeros(al, dtype=float)     # Maximum Intra-Trade equity

#   These arrays are the number of simulation runs
# Max intra-trade drawdown
FC_max_IT_DD = np.zeros(number_forecasts, dtype=float)  
# Trade equity (TWR)
FC_tr_eq = np.zeros(number_forecasts, dtype=float)     

# ------------------------
#   Set up gainer and loser lists
gainer = np.zeros(nrows, dtype=int)
loser = np.zeros(nrows, dtype=int)
i_gainer = 0
i_loser = 0

for i in range(0,nrows-hold_days):
    if (qtC[i+hold_days]>qtC[i]):
        gainer[i_gainer] = i
        i_gainer = i_gainer + 1
    else:
        loser[i_loser] = i
        i_loser = i_loser + 1
number_gainers = i_gainer
number_losers = i_loser

print 'Number Gainers:     %d ' % number_gainers
print 'Number Losers:      %d ' % number_losers
print '            ____Price Distribution____'
hist, bins = np.histogram(qtP.values, bins=100)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width, color='r')
plt.show()
kurt = kurtosis(qtP.values)
sk = skew(qtP.values)
print 'Mean: ', qtP.values.mean(), ' StDev: ',qtP.values.std(), ' Kurtosis: ', kurt, ' Skew: ', sk

X2 = np.sort(qtP)
F2 = np.array(range(qtP.shape[0]), dtype=float)/qtP.shape[0]
print '            ____Cumulative Distribution____'
plt.plot(F2,X2, color='r')
plt.show()
print "Toxic Trades:"
print "TOX05: ", round(stats.scoreatpercentile(X2,5),6), 'Trades:', len(X2)*0.05
print "TOX10: ", round(stats.scoreatpercentile(X2,10),6), 'Trades:', len(X2)*0.10
print "TOX15: ", round(stats.scoreatpercentile(X2,15),6), 'Trades:', len(X2)*0.15
print "TOX20: ", round(stats.scoreatpercentile(X2,20),6), 'Trades:', len(X2)*0.20
print "TOX25: ", round(stats.scoreatpercentile(X2,25),6), 'Trades:', len(X2)*0.25

#entropy
ent = 0
hist = hist[np.nonzero(hist)].astype(float)
for i in hist/sum(hist):
    ent -= i * math.log(i, len(hist))
    #print i,ent
print '\nEntropy:              ', ent

#ADF, Hurst
adf_test(qtC)

#qt_dataSeries = qt.reset_index()
'''
for col in qt_dataSeries:
    if col[:2] == 'ZZ':
        #print 'deleting columns', col
        qt_dataSeries = qt_dataSeries.drop(col,1)

qt_signals = qt.reset_index()
for col in qt_signals:
    if col[:2] != 'ZZ':
        #print 'deleting columns', col
        qt_signals = qt_signals.drop(col,1)
        
#################################################
#setup signals 
#find_long_signals()

model_metrics = init_report()        
CAR25_list_long = []
CAR25_list_short = []
param_grid_long = {'signal': qt_signals.columns, 'direction': ['LONG'], 'trade_signal': [1]}
param_grid_short = {'signal': qt_signals.columns, 'direction': ['SHORT'], 'trade_signal': [-1]}

search_list_long = list(ParameterGrid(param_grid_long))
search_list_short =  list(ParameterGrid(param_grid_short))
total_iter = len(search_list_long+search_list_short)
start_time = time.time()
for i,search in enumerate(search_list_long):
    print '\n', i, ' of ', total_iter, ' elapsed time: ', \
            round(((time.time() - start_time)/60),2), ' minutes'
    metaData = {'ticker':ticker, 'start':start_date, 'end':end_date,\
             'signal':search['signal'], 'rows':qt_dataSeries.shape[0],\
             'accuracy':system_accuracy, 'test_split':0, 'tox_adj':0}
             
    car25 = CAR25_prospector(forecast_horizon, system_accuracy, search['signal'], \
                qt_signals[search['signal']], qt_signals.index.values,\
                qt_dataSeries[' CLOSE'], search['direction'], search['trade_signal'])
            
    model_metrics = update_report_prospector(model_metrics, "PROSPECTOR",\
                    car25, metaData)
                    
    CAR25_list_long.append(car25)
    
for i,search in enumerate(search_list_short):
    print '\n', i+len(search_list_long), ' of ', total_iter, ' elapsed time: ',\
            round(((time.time() - start_time)/60),2), ' minutes'
    metaData = {'ticker':ticker, 'start':start_date, 'end':end_date,\
             'signal':search['signal'], 'rows':qt_dataSeries.shape[0],\
             'accuracy':system_accuracy, 'test_split':0, 'tox_adj':0}
             
    car25 = CAR25_prospector(forecast_horizon, system_accuracy, search['signal'], \
                qt_signals[search['signal']], qt_signals.index.values,\
                qt_dataSeries[' CLOSE'], search['direction'], search['trade_signal'])
            
    model_metrics = update_report_prospector(model_metrics, "PROSPECTOR",\
                    car25, metaData)
                    
    CAR25_list_short.append(car25)
    
CAR25_MAX_long = maxCAR25(CAR25_list_long) 
CAR25_MAX_short = maxCAR25(CAR25_list_short) 
print '\nBest Signal Labels for LONG System'
display_CAR25(CAR25_MAX_long) 
print '\nBest Signal Labels for SHORT System'
display_CAR25(CAR25_MAX_short) 


bestLongSignals = pd.Series(qt_signals[CAR25_MAX_long['Type']], name='Long1 ' + CAR25_MAX_long['Type'])
bestShortSignals = pd.Series(qt_signals[CAR25_MAX_short['Type']], name='Short-1 ' + CAR25_MAX_short['Type'])
'''
#data_to_model = pd.concat([qt_dataSeries,bestLongSignals,bestShortSignals], axis=1)
filename_data = ticker + '_' + str(start_date) + 'to' + str(end_date) + 'to_model.csv'
qt.to_csv(from_fileprep+filename_data)
print '\nSaved file:', from_fileprep+filename_data
#filename_metrics = re.sub(r'[^\w]', '', str(datetime.today())) +ticker+'_prospector.csv'
#model_metrics.sort_values(['CAR25'], ascending=False).to_csv('/media/sf_Python/data/from_fileprep/'+filename_metrics)

print '\nElapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'

##########START DF2############
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

systemName = 'DF_ES_'
#file_path = '/media/sf_Python/data/from_vf_to_df/'
#model_path = '/media/sf_Python/saved_models/DF/'
#save_path_sst = '/media/sf_Python/data/from_df_to_dps/'
#save best models for regime switching here
#save_path_rs = '/media/sf_Python/data/to_RS/'
#scoredModelPath = '/media/sf_Python/FDR_'
#data_type = 'HV'
#filteredDataFile = 'HV_F_ES_TOX25_VotingHard_2001-01-11to2015-12-31_20160107004343.csv'
#data_type = 'LV'
#filteredDataFile = 'LV_F_ES_TOX25_VotingHard_2001-01-11to2015-12-31_20160107004343.csv'
data_type = 'ALL'
#filteredDataFile = 'OHLCV_F_ES_TOX25_VotingHard_v2015-01-01to2017-01-01_20160226054641.csv'
#unfilteredDataFile = 'OHLCV_F_ES_TOX25_VotingHard_v2015-01-01to2017-01-01_20160226054641.csv'
#vf_name = 'None'
#ticker = 'F_ES'
feature_selection = 'Univariate' #RFECV OR Univariate
#vmodel_file = 'VF_20151215193820622423GBCmodelGBCsignalTOX250011497tox_adj1date_end20150331tickerF_SPdate__start20090101Name27dtypeobject.pkl'
signal = 'LongGT0'
#DD95_limit = 0.20
i#nitial_equity = 100000.0
#DF1
wfSteps=[1]
wf_is_periods = [60,90,120,180,250]

#test_split = 0.33 #test split
#iterations = 1 #for sss
tox_adj_proportion = 0
nfeatures = 10

#charts to display in summary
numCharts = 2

#number of sst of systems to save for DPS/RS
nTopSystems = 'ALL'

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


dataSet = pd.concat([qt[' OPEN'],qt[' HIGH'],qt[' LOW'],qt[' CLOSE'],qt[' VOL']], axis=1)
dataSet.columns = ['Open','High','Low','Close','Volume']

nrows = dataSet.shape[0]
print 'Successfully loaded ', nrows,' rows from ', filename_data

RSILookback = 1.5
zScoreLookback = 10
ATRLookback = 5
beLongThreshold = 0.0
DPOLookback = 3    
#takes dc phase 82 rows to ramp up/prime/spit out the first number.   
maxlb = 82


Open = dataSet.Open.values
low = dataSet.Low.values
high = dataSet.High.values
close = dataSet.Close.values
volume = dataSet.Volume.values


  
#CORE indicators
dataSet['Pri_RSI'] = RSI(close,RSILookback)
dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)
dataSet['Pri_ATR'] = zScore(ATR(high,low,close,ATRLookback),zScoreLookback)
dataSet['Pri_ATR_Y1'] = dataSet['Pri_ATR'].shift(1)
dataSet['Pri_ATR_Y2'] = dataSet['Pri_ATR'].shift(2)
dataSet['Pri_ATR_Y3'] = dataSet['Pri_ATR'].shift(3)
dataSet['Rel_ATR'] = zScore(ATR(high,low,close,1)/dataSet['Pri_ATR'].shift(5).values,zScoreLookback)
dataSet['priceChange'] = priceChange(close)
dataSet['priceChangeY1'] = dataSet['priceChange'].shift(1)
dataSet['priceChangeY2'] = dataSet['priceChange'].shift(2)
dataSet['priceChangeY3'] = dataSet['priceChange'].shift(3)
dataSet['pctChangeOpen'] = priceChange(Open)
dataSet['pctChangeLow'] = priceChange(low)
dataSet['pctChangeHigh'] = priceChange(high)
dataSet['Pri_DPO'] = DPO(close,DPOLookback)
#dataSet['Pri_DPO_Y1'] = dataSet['Pri_DPO'].shift(1)
#dataSet['Pri_DPO_Y2'] = dataSet['Pri_DPO'].shift(2)
#dataSet['Pri_DPO_Y3'] = dataSet['Pri_DPO'].shift(3)
dataSet['GARCH'] = zScore(priceChange(garch(dataSet.priceChange).values),zScoreLookback)
dataSet['GARCH_Y1'] = dataSet['GARCH'].shift(1)
dataSet['GARCH_Y2'] = dataSet['GARCH'].shift(2)
#dataSet['GARCH_Y3'] = dataSet['GARCH'].shift(3)
dataSet['autoCor'] = autocorrel(close,3)
dataSet['autoCor_Y1'] = dataSet['autoCor'].shift(1)
dataSet['autoCor_Y2'] = dataSet['autoCor'].shift(2)
dataSet['autoCor_Y3'] = dataSet['autoCor'].shift(3)
dataSet['K_Eff'] = zScore(kaufman_efficiency(close,10),zScoreLookback)# 10 = 2 weeks
dataSet['K_Eff_Y1'] = dataSet['K_Eff'].shift(1)
dataSet['K_Eff_Y2'] = dataSet['K_Eff'].shift(2)
dataSet['K_Eff_Y3'] = dataSet['K_Eff'].shift(3)
dataSet['volSpike'] = zScore(volumeSpike(volume, 5), zScoreLookback)
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


#dataSet['runsScore10'] = runsZScore(dataSet.priceChange,10)
#dataSet['percentUpDays10'] = percentUpDays(dataSet.priceChange,10)
dataSet['mean60_ga'] = pd.rolling_mean(dataSet.priceChange,60)
dataSet['std60_ga'] = pd.rolling_std(dataSet.priceChange,60)
#dataSet['kurt60_ga'] = pd.rolling_kurt(dataSet.priceChange,60)
#dataSet['skew60_ga'] = pd.rolling_skew(dataSet.priceChange,60)


#transform relative to close
#PRICE LEVEL
dataSet['linreg'] = zScore(ta.LINEARREG(close, timeperiod=7)/close,zScoreLookback)
dataSet['tsf'] = zScore(ta.TSF(close, timeperiod=7)/close,zScoreLookback)
dataSet['support'], dataSet['resistance']= ta.MINMAX(close, timeperiod=5)/close
dataSet['support']=zScore(dataSet['support'].values,zScoreLookback)
dataSet['resistance']=zScore(dataSet['resistance'].values,zScoreLookback)
dataSet['bbupper'], dataSet['bbmiddle'], dataSet['bblower'] = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)/close
dataSet['bbupper']=zScore(dataSet['bbupper'].values,zScoreLookback)
dataSet['bbmiddle']=zScore(dataSet['bbmiddle'].values,zScoreLookback)
dataSet['bblower']=zScore(dataSet['bblower'].values,zScoreLookback)

# transform to zs
#MOMENTUM
dataSet['roc'] = zScore(ta.ROC(close, timeperiod=2),20)
dataSet['rocp'] = zScore(ta.ROCP(close, timeperiod=3),20)
dataSet['rocr'] = zScore(ta.ROCR(close, timeperiod=4),20)
dataSet['rocr100'] = zScore(ta.ROCR100(close, timeperiod=5),20)
dataSet['trix'] = zScore(ta.TRIX(close, timeperiod=2),20)
dataSet['ultosc'] = zScore(ta.ULTOSC(high, low, close, timeperiod1=3, timeperiod2=5, timeperiod3=10),20)
dataSet['willr'] = zScore(ta.WILLR(high, low, close, timeperiod=7),20)
dataSet['linreg_slope'] = zScore(ta.LINEARREG_SLOPE(close, timeperiod=7),20)
dataSet['adx'] = zScore(ta.ADX(high, low, close, timeperiod=5),20)
dataSet['adxr'] = zScore(ta.ADXR(high, low, close, timeperiod=3),20)
dataSet['macd'], dataSet['macdsignal'], dataSet['macdhist'] = ta.MACD(close, fastperiod=3, slowperiod=7, signalperiod=3)
dataSet['macd'] = zScore(dataSet['macd'].values,20)
dataSet['macdsignal'] = zScore(dataSet['macdsignal'].values,20)
dataSet['macdhist'] = zScore(dataSet['macdhist'].values,20)
dataSet['minus_di'] = zScore(ta.MINUS_DI(high, low, close, timeperiod=7),20)
dataSet['minus_dm'] = zScore(ta.MINUS_DM(high, low, timeperiod=7),20)
dataSet['cmo'] = zScore(ta.CMO(close, timeperiod=3),20)
dataSet['mom'] = zScore(ta.MOM(close, timeperiod=3),20)
dataSet['plus_di'] = zScore(ta.PLUS_DI(high, low, close, timeperiod=5),20)
dataSet['plus_dm'] = zScore(ta.PLUS_DM(high, low, timeperiod=5),20)
dataSet['ppo'] = zScore(ta.PPO(close, fastperiod=5, slowperiod=10, matype=0),20)
#Volume
#dataSet['ad'] = zScore(ta.AD(high, low, close, volume),20)
#dataSet['adosc'] = zScore(ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10),20)
dataSet['obv'] = zScore(ta.OBV(close, volume),20)
#VOLATILITY
dataSet['atr'] = zScore(ta.ATR(high, low, close, timeperiod=7),20)
dataSet['natr'] = zScore(ta.NATR(high, low, close, timeperiod=10),20)
dataSet['trange'] = zScore(ta.TRANGE(high, low, close),20)
#CYCLE
dataSet['ht_dcperiod'] = zScore(ta.HT_DCPERIOD(close),20)
dataSet['ht_dcphase'] = zScore(ta.HT_DCPHASE(close),20)
dataSet['inphase'], dataSet['quadrature'] = ta.HT_PHASOR(close)
dataSet['inphase']=zScore(dataSet['inphase'].values,20)
dataSet['quadrature']=zScore(dataSet['quadrature'].values,20)

# div 100
#MOMENTUM
dataSet['apo'] = ta.APO(close, fastperiod=3, slowperiod=10, matype=0)/100.0
dataSet['aroondown'], dataSet['aroonup'] = ta.AROON(high, low, timeperiod=7)
dataSet['aroondown']=dataSet['aroondown']/100.0
dataSet['aroonup']=dataSet['aroonup']/100.0
dataSet['aroonosc'] = ta.AROONOSC(high, low, timeperiod=5)/100.0
dataSet['cci'] = ta.CCI(high, low, close, timeperiod=5)/100.0
dataSet['dx'] = ta.DX(high, low, close, timeperiod=5)/100.0
dataSet['mfi'] = ta.MFI(high, low, close, volume, timeperiod=7)/100.0
dataSet['rsi'] = ta.RSI(close, timeperiod=14)/100.0
dataSet['slowk'], dataSet['slowd'] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
dataSet['slowk']=dataSet['slowk']/100.0
dataSet['slowd']=dataSet['slowd']/100.0
dataSet['fastk'], dataSet['fastd'] = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
dataSet['fastk']=dataSet['fastk']/100.0
dataSet['fastd']=dataSet['fastd']/100.0
dataSet['rsifastk'], dataSet['rsifastd'] = ta.STOCHRSI(close, timeperiod=10, fastk_period=5, fastd_period=3, fastd_matype=0)
dataSet['rsifastk']=dataSet['rsifastk']/100.0
dataSet['rsifastd']=dataSet['rsifastd']/100.0

#no transform
dataSet['bop'] = ta.BOP(Open, high, low, close)
#CYCLE
dataSet['sine'], dataSet['leadsine'] = ta.HT_SINE(close)
dataSet['ht_trendmode'] = ta.HT_TRENDMODE(close)

dataSet['gainAhead'] = gainAhead(close)
dataSet['signal'] =  np.where(dataSet.gainAhead>0,1,-1)


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
dataSet['prior_index'] = dataSet.reset_index().index
dataSet.index = dataSet.index.values.astype(str)
dataSet.index = dataSet.index.to_datetime()
dataSet.index.name ='dates'

testFirstYear = dataSet.index[0]
testFinalYear = dataSet.index[wf_is_periods[0]]
validationFirstYear =dataSet.index[wf_is_periods[0]+1]
validationFinalYear =dataSet.index[-1]

#if dt.strptime(testFinalYear, '%Y-%m-%d') >= dt.strptime(validationFirstYear, '%Y-%m-%d'):
#    raise ValueError, 'testFinalYear >= validationFirstYear'

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
            model_metrics, sstDict[runName] = wf_regress_validate(dataSet, dataSet, [m], model_metrics,\
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
