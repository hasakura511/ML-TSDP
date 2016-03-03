# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015

@author: hidemi
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Quandl
from os import listdir
from os.path import isfile, join
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros, runsZScore,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, getToxCutoff2,\
                        percentUpDays
from suztoolz.transform import zigzag as zg
                        
dataPath = 'D:/Dropbox/SharedTSDP/data/ibapi/'
zz_steps = [0.005]
fx_files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

for fx in fx_files:
    data = pd.read_csv(dataPath+fx, index_col='Date')
    ticker = fx[:-3]
    issue = ticker
    
    iterations=10
    RSILookback = 1.5
    zScoreLookback = 10
    ATRLookback = 5
    beLongThreshold = 0.0
    DPOLookback = 3    
    model = LogisticRegression()
    
    open= data.Open.values
    high =data.High.values
    low=data.Low.values
    close=data.Close.values
    
    #volume=np.array([float(x) for x in series[' VOL'].values])
    #oi=series[' OI'].values
    series = data.reset_index() #move date index into col

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




    dataSet = data
    nrows = data.shape[0]
    print nrows
    dataSet['Pri'] = data.Close
    dataSet['Pri_RSI'] = RSI(dataSet.Pri,RSILookback)
    dataSet['Pri_ATR'] = zScore(ATR(data.High,data.Low,data.Close,ATRLookback),
                              zScoreLookback)
    dataSet['Pri_ATR_Y1'] = dataSet['Pri_ATR'].shift(1)
    dataSet['Pri_ATR_Y2'] = dataSet['Pri_ATR'].shift(2)
    dataSet['Pri_ATR_Y3'] = dataSet['Pri_ATR'].shift(3)
    dataSet['priceChange'] = priceChange(dataSet['Pri'])
    dataSet['priceChangeY1'] = dataSet['priceChange'].shift(1)
    dataSet['priceChangeY2'] = dataSet['priceChange'].shift(2)
    dataSet['priceChangeY3'] = dataSet['priceChange'].shift(3)
    dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
    dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
    dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
    dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)

    dataSet['gainAhead'] = gainAhead(dataSet['Pri'])
    dataSet['signal'] = np.where(dataSet.gainAhead>beLongThreshold,1,-1)

    mData = dataSet.drop(['Open','High','Low','Close',
                           'Volume','Pri','gainAhead'],
                            axis=1).dropna()

    #  Select the date range to test
    mmData = mData[:-1]

    datay = mmData.signal
    mmData = mmData.drop(['signal'],axis=1)
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
    print "Learning algorithm is Logistic Regression"
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
    nextSignal = model.predict([mData.drop(['signal'],axis=1).values[-1]])
    print 'Next Signal for',dataSet.index[-1],'is', nextSignal

