# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 01:59:07 2015

@author: hidemi
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

filename = 'F_ES.txt'
issue = filename[:-4]
from_quantiacs = '/media/sf_Python/TSDP/ml/data/tickerData/'
#append price changes from files from this folder
from_quandl = '/media/sf_Python/TSDP/ml/data/quandl/'

#add aux series if difference is less than datapoint_threshold
#datapoint_threshold = -100 
#candle_threshold = .1 #keep candle data if > 10% series


#zigzag steps to check for non-vf, directional trends
zz_steps = [0.005]

#find signals with the highest CAR25
#direction = 'LONG'
start_date = 19981222
end_date = 20160204
years_in_study = int(str(end_date)[:4]) - int(str(start_date)[:4])

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
series[issue] = series[' CLOSE'].pct_change()
for col in series:
    if col == ' P' or col == ' R' or col == ' RINFO':
        #print 'deleting columns', col
        series = series.drop(col,1)
f_count = series[issue].count()

#merge available futures series
aux_files = [ f for f in listdir(from_quantiacs) if isfile(join(from_quantiacs,f)) ]

add_aux = ['F_ED','F_GC','F_DX','F_CL','F_US']

for i,f in enumerate(aux_files):
#    if f != filename:
    if f[:-4] in add_aux:
        #print f
        add_series = pd.read_csv(from_quantiacs+f, error_bad_lines=False, index_col='DATE')
        add_series[f[:-4]] = add_series[' CLOSE']
        for col in add_series:         
            if col != f[:-4]:
                #print 'del', col
                del add_series[col]
        if (add_series.index[0]<=start_date and add_series.index[-1]>=end_date):           
            print 'adding ', f[:-4], add_series.count()[0],'rows to ',issue, f_count, ' rows..'  
            series = pd.merge(series, add_series, left_index=True, right_index=True, how='outer')
        else:
            if add_series.index[0]>start_date:            
                print 'skipping ', f[:-4], ' as ' , add_series.index[0], ' > ', start_date
            if add_series.index[-1]<end_date:
                print 'skipping ', f[:-4], ' as ' , add_series.index[-1], ' < ', end_date

#merge stocks from quandl
aux_files = [ f for f in listdir(from_quandl) if isfile(join(from_quandl,f)) ]
#merge everything
add_aux = [f[:-4] for f in aux_files]

for i,f in enumerate(aux_files):
#    if f != filename:
    if f[:-4] in add_aux:
        #print f
        add_series = pd.read_csv(from_quandl+f, error_bad_lines=False, index_col='Date')
        add_series.index = add_series.index.map(lambda x: x.replace('-','')).astype(int)
        add_series['S_'+f[:-4]] = add_series['Close']
        for col in add_series:         
            if col != 'S_'+f[:-4]:
                #print 'del', col
                del add_series[col]
        if (add_series.index[0]<=start_date and add_series.index[-1]>=end_date):           
            print 'adding ', f[:-4], add_series.count()[0],'rows to ',issue, f_count, ' rows..'  
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
series = series.reset_index() #move date index into col

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

#filename = issue+'_sigcheck.csv'
sig_check = pd.concat([series,zz_signals], axis=1)
#sig_check.dropna().to_csv(filename, index=False)
#print 'Successfully created ', filename

#data_cons = pd.concat([y,series,data,data2,data3,data4], axis=1)

#data_cons.dropna().to_csv(filename[:-4]+'_.csv')

#filename = issue+'_sigcheck.csv'

print '\n\nNew simulation run '
#print 'Testing profit potential for %s positions...\n ' % direction
print 'Issue:             ' + issue
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

qt = sig_check.dropna()
mask = (qt['index'] > start_date) & (qt['index'] <= end_date)
qt = qt.loc[mask]
#print qt.shape
#print qt.head()


#print(qt.loc[mask])

qtC = qt.reset_index()[' CLOSE']
qtP = qt.reset_index()[issue]
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

qt_dataSeries = qt.reset_index()
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
    metaData = {'ticker':issue, 'start':start_date, 'end':end_date,\
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
    metaData = {'ticker':issue, 'start':start_date, 'end':end_date,\
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

data_to_model = pd.concat([qt_dataSeries,bestLongSignals,bestShortSignals], axis=1)
filename_data = issue + '_' + str(start_date) + 'to' + str(end_date) + 'signals.csv'
data_to_model.to_csv('/media/sf_Python/data/from_fileprep/'+filename_data)
print '\nSaved file:', '/media/sf_Python/data/from_fileprep/'+ filename_data
#filename_metrics = re.sub(r'[^\w]', '', str(datetime.today())) +issue+'_prospector.csv'
#model_metrics.sort_values(['CAR25'], ascending=False).to_csv('/media/sf_Python/data/from_fileprep/'+filename_metrics)

print '\nElapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'


