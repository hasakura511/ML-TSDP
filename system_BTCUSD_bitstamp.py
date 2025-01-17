import pandas as pd
import sys
from datetime import datetime
from threading import Event
import numpy as np
import pandas as pd
import time
import json
from time import gmtime, strftime, time, localtime
from io import StringIO

from swigibpy import EWrapper, EPosixClientSocket, Contract
from btapi.get_hist_btcharts import  get_hist_btcharts
from btapi.raw_to_ohlc import raw_to_ohlc_min
from btapi.sort_dates import sort_dates

WAIT_TIME = 30.0
data = pd.DataFrame(columns = ['Date','Open','High','Low','Close','Volume'])
def get_bthist(symbol):
    datestr=strftime("%Y%m%d", localtime())
    data=get_hist_btcharts(symbol);
    dataSet = pd.read_csv(StringIO(data))
    #dataSet.to_csv('./data/btapi/' + symbol + '.csv', index=False)
    dataSet=dataSet.replace([np.inf, -np.inf], np.nan).dropna(how="all")
  
    dataSet=raw_to_ohlc_min(dataSet,'./data/btapi/' + symbol + '.csv')
    #sort_dates('./data/btapi/' + symbol + '.csv','./data/btapi/' + symbol + '.csv')
    dataSet = pd.read_csv('./data/btapi/' + symbol + '.csv', index_col='Date')
    return dataSet
    

data=get_bthist('bitstampUSD').iloc[1000:]

# Simple contract for GOOG
contract = Contract()
contract.exchange = "bitstampUSD"
contract.symbol = "BTC"
contract.secType = "CASH"
contract.currency = "USD"
today = datetime.today()


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
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros, runsZScore,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, getToxCutoff2,\
                        percentUpDays
from suztoolz.display import compareEquity                        

iterations=10
RSILookback = 1.5
zScoreLookback = 10
ATRLookback = 5
beLongThreshold = 0.0
DPOLookback = 3    
model = SVC()
ticker = contract.symbol + contract.currency

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

#  Select the date range to test no label for the last index
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

system="BTCUSD"
data=pd.DataFrame({'Date':dataSet.index[-1], 'Signal':nextSignal}, columns=['Date','Signal'])
#data.to_csv('./data/signals/' + system + '.csv', index=False)
signal=pd.read_csv('./data/signals/' + system + '.csv')
signal=signal.append(data)
if len(sys.argv) > 1:
	signal.to_csv('./data/signals/' + system + '.csv', index=False)
 
 
