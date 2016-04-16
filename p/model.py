import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
from sklearn.qda import QDA
import re
from dateutil import parser
from dateutil.parser import parse
import datetime

import numpy as np
import matplotlib.pyplot as plt
try:
    from matplotlib.finance import quotes_historical_yahoo
except ImportError:
    from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

from os import listdir
from os.path import isfile, join
import re
import sys
import pandas as pd
import features
import classifier
import data
import backtest
import logging
import threading
import seitoolz.signal as signal
import pytz
import seitoolz.bars as bars

portfolio = backtest.MarketIntradayPortfolio()    
nextSignal=0
lastSignal=0
sighist=list()

def count_missing(df):
     return len(df) - df.count()

models=dict()
signals=dict()
def getPredictionFromBestModel(bestdelta, bestlags, fout, cut, start_test, dataSets, parameters):
    global models
    global signals
    """
    returns array of prediction and score from best model.
    """
    algos=['RF','KNN','SVM','ADA','GTB','QDA','GBayes','Voting', 'LDA']
    (X_train, y_train, X_test, y_test)=dataPrep(bestdelta, bestlags, fout, cut, start_test, dataSets, parameters)
    
    for algo in algos:
        if not models.has_key(fout+'_'+algo):
            model = classifier.performClassification(X_train, y_train, X_test, y_test, algo, parameters, fout, False)
            models[fout+'_'+algo]=model
        else:
            model=models[fout+'_'+algo]
        signals[fout+'_'+algo]=[model.predict(X_test), model.score(X_test, y_test)]
    #with open(parameters[0], 'rb') as fin:
    #    model = cPickle.load(fin)        
        
    return signals
    
def dataPrep(maxdelta, maxlag, fout, cut, start_test, dataSets, parameters):
    lags = range(2, maxlag) 
    
    delta = range(2, maxdelta) 
    print 'Delta days accounted: ', max(delta)
    datasets = features.applyRollMeanDelayedReturns(dataSets, delta)
    finance = features.mergeDataframes(datasets, 6, cut)
    print 'Size of data frame: ', finance.shape
    print 'Number of NaN after merging: ', count_missing(finance)
    finance = finance.interpolate(method='linear')
    print 'Number of NaN after time interpolation: ', count_missing(finance)
    finance = finance.fillna(finance.mean())
    print 'Number of NaN after mean interpolation: ', count_missing(finance)    
    finance = features.applyTimeLag(finance, lags, delta)
    print 'Number of NaN after temporal shifting: ', count_missing(finance)
    print 'Size of data frame after feature creation: ', finance.shape
    X_train, y_train, X_test, y_test  = classifier.prepareDataForClassification(finance, start_test)
    return (X_train, y_train, X_test, y_test)
    
def performFeatureSelection(maxdeltas, maxlags, fout, cut, start_test, initDataSets, savemodel, method, folds, parameters):
    """
    Performs Feature selection for a specific algorithm
    """
    
    for maxlag in range(3, maxlags + 2):
        lags = range(2, maxlag) 
        print ''
        print '============================================================='
        print 'Maximum time lag applied', max(lags)
        print ''
        for maxdelta in range(3, maxdeltas + 2):
            datasets=initDataSets.copy()
            delta = range(2, maxdelta) 
            print 'Delta days accounted: ', max(delta)
            datasets = features.applyRollMeanDelayedReturns(datasets, delta)
            finance = features.mergeDataframes(datasets, 6, cut)
            print 'Size of data frame: ', finance.shape
            print 'Number of NaN after merging: ', count_missing(finance)
            finance = finance.interpolate(method='linear')
            print 'Number of NaN after time interpolation: ', count_missing(finance)
            finance = finance.fillna(finance.mean())
            print 'Number of NaN after mean interpolation: ', count_missing(finance)    
            finance = features.applyTimeLag(finance, lags, delta)
            print 'Number of NaN after temporal shifting: ', count_missing(finance)
            print 'Size of data frame after feature creation: ', finance.shape
            X_train, y_train, X_test, y_test  = classifier.prepareDataForClassification(finance, start_test)
            #accuracies = classifier.performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel)
            #print accuracies
            print performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
            print ''

def performCV(X_train, y_train, number_folds, algorithm, parameters, fout, savemodel):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the ML algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.
    """

    print 'Parameters --------------------------------> ', parameters
    print 'Size train set: ', X_train.shape
    
    # k is the size of each fold. It is computed dividing the number of 
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print 'Size of each fold: ', k
    
    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    accuracies = np.zeros(number_folds-1)

    # loop from the first 2 folds to the total number of folds    
    for i in range(2, number_folds + 1):
        print ''
        
        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out 
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second), 
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds 
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i
        
        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        print 'Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) 
        
        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size 
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # .... 
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        print 'Size of train + test: ', X.shape # the size of the dataframe is going to be k*i

        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]
        
        # i starts from 2 so the zeroth element in accuracies array is i-2. performClassification() is a function which takes care of a classification problem. This is only an example and you can replace this function with whatever ML approach you need.
        accuracies[i-2] = classifier.performClassification(X_trainFolds, y_trainFolds, X_testFold, y_testFold, algorithm, parameters, fout, savemodel)
        
        # example with i = 4:
        #      Accuracy on fold         4     :    0.85423
        print 'Accuracy on fold ' + str(i) + ': ', accuracies[i-2]
    
    # the function returns the mean of the accuracy on the n-1 folds    
    return accuracies.mean()

def get_signal(lookback, portfolio, argv):
    global nextSignal
    global lastSignal
    global start_period
    global start_test
    global symbol
    global file
    global path_datasets
    global name
    global folds
    global bestModel
    global interval
    global dataSets
    global sighist
    lastSignal=dict()
    nextSignal=dict()
    sighist=dict()
    bestModel=''
    #performCV(X_train, y_train, 10, 'QDA', [])
    start_period = datetime.datetime(2015,12,15)  
    start_test = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=lookback*2+14)  
    symbol = 'EURJPY'
    file='30m_EURJPY'
    path_datasets='./data/from_IB/'
    name = './p/data/' + file + '.csv'
    folds=10
    interval='30m_'
    
    if len(argv) > 1 and argv[1] == '1':
        ############## idx ##############    
        symbol = '^GSPC'
        file='idx_^GSPC'
        start_period = datetime.datetime(1995,1,15)  
        start_test = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=lookback*2+14)  
        end_period = datetime.datetime.now()        
        interval='idx_'
        path_datasets='./p/data/'
        name = './p/data/' + file + '.csv'
        if len(argv) > 2 and argv[2] == '2':
            (out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia)=data.getStockDataFromWeb(symbol, name, path_datasets, str(start_period), str(end_period))
        ############## idx ##############  
    elif len(argv) > 1 and argv[1] == '2':
        ############## idx ##############    
        symbol = 'EURJPY'
        if len(argv) > 4:
            symbol=argv[4]
        interval='30m_'        
        file=interval + symbol
        bar=bars.get_bar(file)
        date=parse(bar.index[-1])
        #start_period = datetime.datetime(2015,12,15)  
        start_test = date - datetime.timedelta(minutes=lookback*30*2+72*30)  
        start_period = start_test - datetime.timedelta(hours=2000)
        path_datasets='./data/from_IB/'
        name = './p/data/' + file + '.csv'
        folds=10
        ############## idx ##############  
    elif len(argv) > 1 and argv[1] == '3':
        ############## idx ##############    
        path_datasets='./quantiacs/tickerData/'
        symbol = 'ES'
        file='F_ES'
        start_period = datetime.datetime(1995,1,15)  
        start_test = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=lookback*2+14)  
        end_period = datetime.datetime.now()        
        interval='F_'
        
        name = './p/data/' + file + '.csv'
        #if len(argv) > 2 and argv[2] == '2':
            #(out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia)=data.getStockDataFromWeb(symbol, name, path_datasets, str(start_period), str(end_period))
        ############## idx ##############  
    elif len(argv) > 1 and argv[1] == '4':
        symbol = 'EURJPY'
        if len(argv) > 4:
            symbol=argv[4]
        interval='1h_'        
        file=interval + symbol
        bar=bars.get_bar(file)
        date=parse(bar.index[-1])
        #start_period = datetime.datetime(2015,07,15)  
        start_test = date - datetime.timedelta(hours=lookback*2+72)  
        start_period = start_test - datetime.timedelta(hours=4000)
        path_datasets='./data/from_IB/'
        name = './p/data/' + file + '.csv'
        folds=10
    return next_signal(lookback, portfolio, argv)

def next_signal(lookback, portfolio, argv):
    global nextSignal
    global lastSignal
    global start_period
    global start_test
    global symbol
    global file
    global path_datasets
    global name
    global folds
    global bestModel
    global interval
    global dataSets
    global sighist
    bestModelFile='./p/params/bestModel.pickle'
    parameters=list()
    parameters.append(bestModelFile)    
    parameters.append(interval)
    parameters.append(lookback)
    bData=list()
    dataSets = data.loadDatasets(path_datasets, file, parameters)
    bData=dataSets
    print 'Training with sample up to: ' + str(start_test)
    print 'Creating Backtest Signal Up To: ' + str(bData[0].index[-2])
    print 'Last Open: ',str(bData[0]['Open_Out'][-2]),  ' Last Close: ', bData[0]['Close_Out'][-2], ' Providing Look Future Data: Open ',bData[0]['Open_Out'][-1],' Close ',bData[0]['Close_Out'][-1]
    if len(argv) > 2 and argv[2] == '1':
        prediction = performFeatureSelection(9, 9, file, start_period, start_test, bData, True, 'RF', folds, parameters)    
    signals=getPredictionFromBestModel(9, 9, file, start_period, start_test, bData, parameters)
    bestalgo=dict()    
    for algo in signals.keys():
        (prediction, accuracy)=signals[algo]
        if nextSignal.has_key(algo):
            lastSignal[algo]=nextSignal[algo]
        else:
            lastSignal[algo]=0
        nextSignal[algo]=prediction[-1]
        if nextSignal[algo] == 0:
            nextSignal[algo]=-1
        bestalgo[algo]=[prediction, accuracy]
        if len(bestModel) == 0 or signals[algo][1] > signals[bestModel][1]:
            bestModel=algo
        print algo, ' Accuracy ', accuracy
        print algo, ' Next Signal: ' + str(nextSignal[algo])
        logging.info(algo + ' Next Signal: ' + str(nextSignal[algo]))
    
    print 'Best Model: ',bestModel, ' Accuracy: ', signals[bestModel][1]
    #if len(argv) > 2 and argv[2] == '2':
    #        return nextSignal[bestModel]
    
    # dataframe of Historical Price
    bars=data.get_quote(path_datasets, file, '', False, parameters)
    # subset of the data corresponding to test set
    #if parameters[2] > 0:
    #    bars=bars.iloc[:-parameters[2]]  
    
    bars.index=bars.index.to_pydatetime()
    #bars = bars.ix[start_test:]
    
    bars = bars[-(len(prediction[:-1])):]
    bars=bars.sort_index()
    print 'Backtesting with data feed up to: ' + str(bars.index[-1])
    # initialize empty dataframe indexed as the bars. There's going to be perfect match between dates in bars and signals 
    signals = pd.DataFrame(index=bars.index)
    algos=bestalgo.keys()
    for algo in algos:
        # initialize signals.signal column to zero
        signals[algo] = 0.0
        (prediction, accuracy)=bestalgo[algo]
        # copying into signals.signal column results of prediction
        print algo, ' with ', bars.shape[0],' bars ', len(prediction[:-1]), 'predictions'
        if sighist.has_key(algo):
            print 'Last Signal' + str(lastSignal[algo])
            sighist[algo]=np.append(sighist[algo], lastSignal[algo])
        else:
            sighist[algo]=np.array(prediction[:-1])
            
        if len(bars.index) < len(sighist[algo]):
            print 'Bar Length: ', len(bars.index),' Longer than Sig Length:',len(sighist[algo]), 'Adjusting sighist'
            sighist[algo]=sighist[algo][-len(bars.index):]
        print algo, ' Bar Length: ', len(bars.index),' Sig Length:',len(sighist[algo])
        signals[algo] = sighist[algo]
         
        # replace the zeros with -1 (new encoding for Down day)
        signals.ix[signals[algo] == 0, algo] = -1
        signals[algo+'_pos'] = signals[algo].diff()     
    
    
    print 'Last Bar: ' + str(bars.iloc[-1]['Close'])
    logging.info('Last Bar: ' + str(bars.iloc[-1]['Close']))
    # compute the difference between consecutive entries in signals.signal. As
    # signals.signal was an array of 1 and -1 return signals.positions will 
    # be an array of 0s and 2s.
     
    # calling portfolio evaluation on signals (predicted returns) and bars 
    # (actual returns)
    portfolio.setInit(symbol, bars, signals, algos, nextSignal, lastSignal, parameters)
    
    # backtesting the portfolio and generating returns on top of that 
    (portfolioData, returns, ranking)=portfolio.backtest_portfolio()
    count=len(ranking)
    for [algo,equity] in ranking:
        print algo, '#',count, ' with returns of: $', equity, ' Next Signal: ', nextSignal[algo]
        count = count - 1
    (bestModel,bestEquity) =ranking[-1]
    print 'Best Model: ',bestModel,' Next Signal:',nextSignal[algo]
    if len(argv) > 2 and argv[2] == '2':
        return nextSignal[bestModel]
    
    print 'Backtesting Complete'
    return nextSignal[bestModel]
    
    
def start_lookback(lookback, argv):
     for num in range(0,lookback+1):
        num=lookback-num
        if num == lookback:
            get_signal(num, portfolio, argv)
        else:
            next_signal(num,portfolio, argv)
    