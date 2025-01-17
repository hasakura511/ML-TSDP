import pandas as pd
import sys
from datetime import datetime
from threading import Event

from swigibpy import EWrapper, EPosixClientSocket, Contract

start = '20160222  08:00:00'
WAIT_TIME = 30.0
data = pd.DataFrame(columns = ['Open','High','Low','Close','Volume'])
#p_open=[];
#p_high=[];
#p_low=[];
#p_close=[];
#p_volume=[];
#p_chg=[];

###


class HistoricalDataExample(EWrapper):
    '''Callback object passed to TWS, these functions will be called directly
    by TWS.
    '''

    def __init__(self):
        super(HistoricalDataExample, self).__init__()
        self.got_history = Event()

    def orderStatus(self, id, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFilledPrice, clientId, whyHeld):
        pass

    def openOrder(self, orderID, contract, order, orderState):
        pass

    def nextValidId(self, orderId):
        '''Always called by TWS but not relevant for our example'''
        pass

    def openOrderEnd(self):
        '''Always called by TWS but not relevant for our example'''
        pass

    def managedAccounts(self, openOrderEnd):
        '''Called by TWS but not relevant for our example'''
        pass

    def historicalData(self, reqId, date, open, high,
                       low, close, volume,
                       barCount, WAP, hasGaps):

        if date[:8] == 'finished':
            print("History request complete")
            self.got_history.set()
        else:
	    #chg=0;
	    #chgpt=0;
	    #if len(p_close) > 0:
	    #	chgpt=close-p_close[-1];
		#chg=chgpt/p_close[-1];
           
        #    p_open.append(open);
        #    p_high.append(high);
        #    p_low.append(low);
        #    p_close.append(close);
        #    p_volume.append(volume);
	    #date = datetime.strptime(date, "%Y%m%d").strftime("%d %b %Y")
            data.loc[date] = [open,high,low,close,volume]
            #print "History %s - Open: %s, High: %s, Low: %s, Close: %s, Volume: %d"\
            #           % (date, open, high, low, close, volume)

            #print(("History %s - Open: %s, High: %s, Low: %s, Close: "
            #       "%s, Volume: %d, Change: %s, Net: %s") % (date, open, high, low, close, volume, chgpt, chg));


# Instantiate our callback object
callback = HistoricalDataExample()

# Instantiate a socket object, allowing us to call TWS directly. Pass our
# callback object so TWS can respond.

tws = EPosixClientSocket(callback)
#tws = EPosixClientSocket(callback, reconnect_auto=True)
# Connect to tws running on localhost
if not tws.eConnect("", 7496, 111):
    raise RuntimeError('Failed to connect to TWS')

# Simple contract for GOOG
contract = Contract()
contract.exchange = "IDEALPRO"
contract.symbol = "USD"
contract.secType = "CASH"
contract.currency = "CHF"
today = datetime.today()

print("Requesting historical data for %s" % contract.symbol)

# Request some historical data.
tws.reqHistoricalData(
    1,                                         # tickerId,
    contract,                                   # contract,
    today.strftime("%Y%m%d %H:%M:%S %Z"),       # endDateTime,
    "1 M",                                      # durationStr,
    "1 hour",                                    # barSizeSetting,
    "ASK",                                   # whatToShow,
    1,                                          # useRTH,
    1                                          # formatDate
)

print("\n====================================================================")
print(" History requested, waiting %ds for TWS responses" % WAIT_TIME)
print("====================================================================\n")


try:
    callback.got_history.wait(timeout=WAIT_TIME)
except KeyboardInterrupt:
    pass
finally:
    if not callback.got_history.is_set():
        print('Failed to get history within %d seconds' % WAIT_TIME)

    print("\nDisconnecting...")
    tws.eDisconnect()

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
'''
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)        
model = VotingClassifier(estimators=[\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             ("QDA", QuadraticDiscriminantAnalysis()),\
             ("GNBayes",GaussianNB()),\
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
             #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             ("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
             ("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             ("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                ], voting='hard', weights=None)
'''
model = GaussianNB()               
ticker = contract.symbol + contract.currency

dataSet = data
nrows = data.shape[0]
print nrows
dataSet['Pri'] = data.Close
dataSet['Pri_RSI'] = RSI(dataSet.Pri,RSILookback)
dataSet['Pri_DPO'] = DPO(dataSet.Close,DPOLookback)
dataSet['Pri_ATR'] = zScore(ATR(data.High,data.Low,data.Close,ATRLookback),
                          zScoreLookback)
dataSet['Pri_ATR_Y1'] = dataSet['Pri_ATR'].shift(1)
dataSet['Pri_ATR_Y2'] = dataSet['Pri_ATR'].shift(2)
dataSet['Pri_ATR_Y3'] = dataSet['Pri_ATR'].shift(3)
dataSet['priceChange'] = priceChange(dataSet['Pri'])
dataSet['priceChangeY1'] = dataSet['priceChange'].shift(1)
dataSet['priceChangeY2'] = dataSet['priceChange'].shift(2)
dataSet['priceChangeY3'] = dataSet['priceChange'].shift(3)
dataSet['Pri_DPO_Y1'] = dataSet['Pri_DPO'].shift(1)
dataSet['Pri_DPO_Y2'] = dataSet['Pri_DPO'].shift(2)
dataSet['Pri_DPO_Y3'] = dataSet['Pri_DPO'].shift(3)
dataSet['Pri_DPO_Y4'] = dataSet['Pri_DPO'].shift(4)

dataSet['gainAhead'] = gainAhead(dataSet['Pri'])
dataSet['signal'] = np.where(dataSet.gainAhead>beLongThreshold,1,-1)

mData = dataSet.drop(['Open','High','Low','Close',
                       'Volume','Pri','gainAhead'],
                        axis=1).dropna()

#  Select the date range to test no label for the last index
mmData = mData.ix[start:mData.index[-2]]

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
model.fit(dX, dy)
ypred = model.predict(dX)
sst= pd.concat([dataSet['gainAhead'].ix[datay.index], \
            pd.Series(data=ypred,index=datay.index, name='signals')],axis=1)
sst.index=sst.index.astype(str).to_datetime()
if len(sys.argv)==1:
    compareEquity(sst, ticker)

nextSignal = model.predict([mData.drop(['signal'],axis=1).values[-1]])
print 'Next Signal for',dataSet.index[-1],'is', nextSignal

system="USDCHF"
data=pd.DataFrame({'Date':dataSet.index[-1], 'Signal':nextSignal}, columns=['Date','Signal'])
#data.to_csv('./data/signals/' + system + '.csv', index=False)
signal=pd.read_csv('./data/signals/' + system + '.csv')
signal=signal.append(data)
signal.to_csv('./data/signals/' + system + '.csv', index=False)