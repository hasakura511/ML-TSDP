import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
from pytz import timezone
from datetime import datetime as dt
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import datetime
from datetime import datetime as dt
from scipy import stats
from scipy.stats import kurtosis, skew
import statsmodels.tsa.stattools as ts
from sklearn.preprocessing import scale, robust_scale, minmax_scale
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,\
                            log_loss,precision_score,recall_score, roc_auc_score,\
                            confusion_matrix, hamming_loss, jaccard_similarity_score,\
                            zero_one_loss

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

#other
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV

#suztoolz
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve,\
                         directional_scoring, compareEquity, describeDistribution
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df,\
                            maxCAR25, wf_regress_validate2, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
from suztoolz.data import getDataFromIB
SST = bestBothDPS[['signals','gainAhead']]
title = runName
leverage = bestBothDPS.safef.values

#def calcEquity_df(SST, title, leverage=1.0):
initialEquity = 1.0
nrows = SST.gainAhead.shape[0]
#signalCounts = SST.signals.shape[0]
print '\nThere are %0.f signal counts' % nrows
if 1 in SST.signals.value_counts():
    print SST.signals.value_counts()[1], 'beLong Signals',
if -1 in SST.signals.value_counts():
    print SST.signals.value_counts()[-1], 'beShort Signals',
if 0 in SST.signals.value_counts():
    print SST.signals.value_counts()[0], 'beFlat Signals',
    
equityCurves = {}
for trade in ['l','s','b']:       
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    num_days = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numDays')
    equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
    maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
    drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
    maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
    safef = pd.Series(data=leverage,index=range(0,len(SST.index)),name='safef')

    for i in range(0,len(SST.index)):
        if i == 0:
            equity[i] = initialEquity
            trades[i] = 0.0
            num_days[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0

        else:
            if trade=='l':
                if (SST.signals[i-1] > 0):
                    trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                    num_days[i] = num_days[i-1] + 1 
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])

                    #print i, SST.signals[i], trades[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
                else:
                    trades[i] = 0.0
                    num_days[i] = num_days[i-1]
                    equity[i] = equity[i-1]
                    maxEquity[i] = maxEquity[i-1]
                    drawdown[i] = drawdown[i-1]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
            elif trade=='s':
                if (SST.signals[i-1] < 0):
                    trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                    num_days[i] = num_days[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    trades[i] = 0.0
                    num_days[i] = num_days[i-1]
                    equity[i] = equity[i-1]
                    maxEquity[i] = maxEquity[i-1]
                    drawdown[i] = drawdown[i-1]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
            else:
                if (SST.signals[i-1] > 0):
                    trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                    num_days[i] = num_days[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                elif (SST.signals[i-1] < 0):
                    trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                    num_days[i] = num_days[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    trades[i] = 0.0
                    num_days[i] = num_days[i-1]
                    equity[i] = equity[i-1]
                    maxEquity[i] = maxEquity[i-1]
                    drawdown[i] = drawdown[i-1]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                    
    SSTcopy = SST.copy(deep=True)
    if trade =='l':
        #changeIndex = SSTcopy.signals[SST.signals==-1].index
        SSTcopy.loc[SST.signals==-1,'signals']=0
    elif trade =='s':
        #changeIndex = SSTcopy.signals[SST.signals==1].index
        SSTcopy.loc[SST.signals==1,'signals']=0
        
    equityCurves[trade] = pd.concat([SSTcopy.reset_index(), safef, trades, num_days, equity,maxEquity,drawdown,maxDD], axis =1)

#  Compute cumulative equity for all days (buy and hold)   
trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
num_days = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numDays')
equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
safef = pd.Series(data=1.0,index=range(0,len(SST.index)),name='safef')
for i in range(0,len(SST.index)):
    if i == 0:
        equity[i] = initialEquity
        trades[i] = 0.0
        num_days[i] = 0.0
        maxEquity[i] = initialEquity
        drawdown[i] = 0.0
        maxDD[i] = 0.0
    else:
        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
        num_days[i] = num_days[i-1] + 1 
        equity[i] = equity[i-1] + trades[i]
        maxEquity[i] = max(equity[i],maxEquity[i-1])
        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
        maxDD[i] = max(drawdown[i],maxDD[i-1])          
SSTcopy.loc[SST.signals==-1,'signals']=1
SSTcopy.loc[SST.signals==0,'signals']=1

equityCurves['buyHold'] = pd.concat([SSTcopy.reset_index(), safef, trades, num_days, equity,maxEquity,drawdown,maxDD], axis =1)

if not SST.index.to_datetime()[0].time() and not SST.index.to_datetime()[1].time():
    barSize = '1 day'
else:
    barSize = '1 min'
    
#plt.close('all')
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(8,7))
#plt.subplot(2,1,1)
ind = np.arange(SST.shape[0])
ax1.plot(ind, equityCurves['l'].equity,label="Long 1 Signals",color='b')
ax1.plot(ind, equityCurves['s'].equity,label="Short -1 Signals",color='r')
ax1.plot(ind, equityCurves['b'].equity,label="Long & Short",color='g')
ax1.plot(ind, equityCurves['buyHold'].equity,label="BuyHold",ls='--',color='c')
#fig, ax = plt.subplots(2)
#plt.subplot(2,1,2)
ax2.plot(ind, -equityCurves['l'].drawdown,label="Long 1 Signals",color='b')
ax2.plot(ind, -equityCurves['s'].drawdown,label="Short -1 Signals",color='r')
ax2.plot(ind, -equityCurves['b'].drawdown,label="Long & Short",color='g')
ax2.plot(ind, -equityCurves['buyHold'].drawdown,label="BuyHold",ls='--',color='c')

if barSize != '1 day' :
    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
        return SST.index[thisind].strftime("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    
else:
    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
        return SST.index[thisind].strftime("%Y-%m-%d")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    
# rotate and align the tick labels so they look better
fig.autofmt_xdate()

# use a more precise date string for the x axis locations in the
# toolbar

fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax1.set_title(title)
ax1.set_ylabel("TWR")
ax1.legend(loc="best")
ax2.set_ylabel("Drawdown")   
plt.show()
shortTrades, longTrades = numberZeros(SST.signals)
allTrades = shortTrades+ longTrades
print '\nValidation Period from', SST.index[0],'to',SST.index[-1]
print 'TWR for Buy & Hold is %0.3f, %i days, maxDD %0.3f' %\
            (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
print 'TWR for %i beLong trades is %0.3f, maxDD %0.3f' %\
            (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
print 'TWR for %i beShort trades is %0.3f, maxDD %0.3f' %\
            (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
print 'TWR for %i beLong and beShort trades is %0.3f, maxDD %0.3f' %\
            (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
print 'SAFEf:', equityCurves['b'].safef.mean()

SST_equity = equityCurves['b']
#    if 'dates' in SST_equity:
#        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['dates'])).drop(['dates'], axis=1)
#    else:
#        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['index'])).drop(['index'], axis=1)