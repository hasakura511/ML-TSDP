# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015
v1.10
added param load


@author: hidemi
"""
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Quandl
from os import listdir
from os.path import isfile, join
from datetime import datetime as dt
import datetime 

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                        BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
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
import pandas as pd
import sys
from datetime import datetime
from threading import Event

from swigibpy import EWrapper, EPosixClientSocket, Contract


start_time = time.time()
bestParamsPath = './data/params/'
signalPath = './data/signals/'
livePairs = [
                'AUDUSD',\
                'EURUSD',\
                'GBPUSD',\
                'USDCAD',\
                'USDCHF',\
                'USDJPY',\
                'EURJPY',
                ]
                
currencyPairs = [
                'AUDUSD',\
                'EURUSD',\
                'GBPUSD',\
                'USDCAD',\
                'USDCHF',\
                'USDJPY',\
                'EURJPY'
                ]
                
WAIT_TIME = 30.0

#system parameters
#version = 'v2.21C'
filterName = 'DF1'
data_type = 'ALL'

# save 0 to signal file if pair is turned off
for pair in currencyPairs:
    if pair not in livePairs:
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if pair + '.csv' not in files:
            pass
        else:
            signalFile=pd.read_csv(signalPath+ pair + '.csv', parse_dates=['Date'])
            offline = signalFile.iloc[-1].copy(deep=True)
            offline.dates = pd.to_datetime(dt.now().replace(second=0, microsecond=0))
            offline.signals = 0
            #offline.gainAhead =0
            #offline.prior_index=0
            #offline.safef=0
            #offline.CAR25=0
            #offline.dd95 = 0
            #offline.ddTol=0
            #offline.system = 'Offline'
            signalFile=signalFile.append(offline)
            signalFile.to_csv(signalPath + pair + '.csv', index=False)
            #sys.exit("Offline Mode: "+sys.argv[0])
############################################################
currencyPairsDict = {}
barSizeSetting='1 min'
dataPath = './data/from_IB/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

for pair in currencyPairs:    
    if barSizeSetting+'_'+pair+'.csv' in files:
        data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)
        currencyPairsDict[pair] = data
        
print 'Successfully Retrieved Data.'
###########################################################
for ticker in livePairs:     
    symbol=ticker[0:3]
    currency=ticker[3:6]
    bestModel = pd.read_csv(bestParamsPath+ticker+'.csv').iloc[-1]
    data = currencyPairsDict[ticker]
    print 'received', data.shape[0], 'rows of data'                

    #Model Parameters
    signal = bestModel.signal    
    #perturbDataPct = 0.0002
    #longMemory =  False
    #iterations=1
    #input_signal = 1
    #feature_selection = 'None' #RFECV OR Univariate
    #wfSteps=[1]
    wf_is_period = bestModel.rows
    #wf_is_periods = [100]
    #tox_adj_proportion = 0
    #nfeatures = 10

    iterations=10
    RSILookback = bestModel.RSILookback
    zScoreLookback = bestModel.zScoreLookback
    ATRLookback = bestModel.ATRLookback
    beLongThreshold = bestModel.beLongThreshold
    DPOLookback = bestModel.DPOLookback
    ACLookback = bestModel.ACLookback
    CCLookback = bestModel.CCLookback
    rStochLookback = bestModel.rStochLookback
    statsLookback = bestModel.statsLookback
    ROCLookback = bestModel.ROCLookback
    model = eval(bestModel.params)
    ticker = symbol + currency

    dataSet = data
    #nrows = data.shape[0]
    #print nrows

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
    if signal[:2] =='ZZ':
        zz_step = [float(x) for x in signal.split()[1].split(',')]
        zz_signals = pd.DataFrame(index = dataSet.index)
        print 'Creating Signal labels..',
        print signal
        zz_signals[signal] = zg(dataSet.Close, zz_step[0], zz_step[-1]).pivots_to_modes()
        dataSet['signal'] = zz_signals[signal].shift(-1).fillna(0)
    else:
        dataSet['signal'] =  np.where(dataSet.gainAhead>0,1,-1)
        
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

    mData = dataSet.drop(['Open','High','Low','Close',
                           'Volume','prior_index','gainAhead'],
                            axis=1).dropna()

    #  Select the date range to test no label for the last index
    mmData = mData.iloc[-wf_is_period:-1]

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
    '''
    model = SVC()
    from sklearn.grid_search import GridSearchCV
    Crange = np.logspace(-2,2,40)
    grid = GridSearchCV(model, param_grid={'C':Crange},scoring='accuracy',cv=5)
    grid.fit(dX,dy)
    grid.best_params_
    score = [g[1] for g in grid.grid_scores_]
    score
    plt.semilogx(Crange,scores)
    plt.semilogx(Crange,score)
    '''
    model.fit(dX, dy)
    ypred = model.predict(dX)
    sst= pd.concat([dataSet['gainAhead'].ix[datay.index], \
                pd.Series(data=ypred,index=datay.index, name='signals')],axis=1)
    sst.index=sst.index.astype(str).to_datetime()
    
    if len(sys.argv)==1:
        compareEquity(sst, ticker)

    nextSignal = model.predict([mData.drop(['signal'],axis=1).values[-1]])
    print 'Next Signal for',dataSet.index[-1],'is', nextSignal

    signal_df=pd.DataFrame({'Date':dataSet.index[-1], 'Signal':nextSignal}, columns=['Date','Signal'])
    #signal.to_csv('./signal/signalss/' + ticker + '.csv', index=False)
    signalFile=pd.read_csv(signalPath + ticker + '.csv')
    signalFile=signalFile.append(signal_df)
    if len(sys.argv) > 1:
        signalFile.to_csv(signalPath + ticker + '.csv', index=False)
     
     
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
