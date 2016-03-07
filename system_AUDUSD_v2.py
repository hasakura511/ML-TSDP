
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015

@author: hidemi
"""
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
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data
from suztoolz.data import getDataFromIB

start_time = time.time()
'''
debug = False

if len(sys.argv)==1:
    debug=True
    
if debug:
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    scorePath = './debug/scored_metrics_'
    equityStatsSavePath = './debug/'
    signalPath =  './debug/'
else:
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/' 
'''

debug=False

if len(sys.argv)==1:
    debug=True

if debug:
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath =  'C:/Users/Hidemi/Desktop/Python/'
else:
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/' 


#system parameters
version = 'v2'
filterName = 'DF1'
data_type = 'ALL'
signal = 'LongGT0'

#data Parameters
cycles = 3
exchange='IDEALPRO'
symbol='AUD'
secType='CASH'
currency='USD'
endDateTime=dt.now(timezone('US/Eastern'))
durationStr='1 M'
barSizeSetting='1 hour'
whatToShow='MIDPOINT'
ticker = symbol + currency

#Model Parameters
perturbDataPct = 0.0002
longMemory =  False
iterations=1
input_signal = 1
feature_selection = 'None' #RFECV OR Univariate
wfSteps=[1]
wf_is_periods = [100,200,300,500]
#wf_is_periods = [100]
tox_adj_proportion = 0
nfeatures = 10

#feature Parameters
RSILookback = 1.5
zScoreLookback = 10
ATRLookback = 5
beLongThreshold = 0.0
DPOLookback = 10
ACLookback = 10

#DPS parameters
windowLengths = [12,24]
maxLeverage = [10,20]
PRT={}
PRT['DD95_limit'] = 0.05
PRT['tailRiskPct'] = 95
PRT['initial_equity'] = 1.0
PRT['horizon'] = 250
PRT['maxLeverage'] = 2
#CAR25_threshold=-np.inf
CAR25_threshold=0

#model selection
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
        #("KNNu", KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)),\
        #("KNNd", KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)),\
        #("KNeighborsRegressor-u,p1,n15", KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=1, metric='minkowski', metric_params=None, n_jobs=-1)),\
        #("RadiusNeighborsRegressor",RadiusNeighborsRegressor()),\
        #("LinearSVR", LinearSVR()),\
        #("rbf.01SVR",SVR(kernel='rbf', C=0.1, gamma=.01)),\
        ("rbf1SVR",SVR(kernel='rbf', C=1, gamma=0.1)),\
        #("rbf10SVR",SVR(kernel='rbf', C=10, gamma=0.1)),\
        #("rbf100SVR",SVR(kernel='rbf', C=100, gamma=0.1)),\
        #("rbf1000SVR",SVR(kernel='rbf', C=1000, gamma=0.1)),\
        #("rbf100SVR",SVR(kernel='rbf', C=1e3, gamma=0.1)),\
        #("polySVR",SVR(kernel='poly', C=1e3, degree=2)),\
         ]


############################################################
for i in range(0,cycles):
    if i ==0:
        getHistLoop = [endDateTime]
    else:
        getHistLoop.insert(0,(getHistLoop[0]-datetime.timedelta(365/12)))

getHistLoop =   [x.strftime("%Y%m%d %H:%M:%S %Z") for x in getHistLoop]

brokerData = {}
brokerData =  {'port':7496, 'client_id':101,\
                     'tickerId':1, 'exchange':exchange,'symbol':symbol,'secType':secType,'currency':currency,\
                     'endDateTime':endDateTime, 'durationStr':durationStr,'barSizeSetting':barSizeSetting,\
                     'whatToShow':whatToShow, 'useRTH':1, 'formatDate':1
                      }
                      
data = pd.DataFrame()                   
for date in getHistLoop:
    brokerData['client_id']=random.randint(100,1000)
    data = pd.concat([data,getDataFromIB(brokerData, date)],axis=0)
###########################################################
print 'Successfully Retrieved Data.  Begin Preprocessing...'
#perturb dataSet
if perturbData:
    #dataSet['OPEN'] = perturb_data(dataSet['OPEN'].values,perturbDataLookback)
    #dataSet['HIGH']= perturb_data(dataSet['HIGH'].values,perturbDataLookback)
    #dataSet['LOW']= perturb_data(dataSet['LOW'].values,perturbDataLookback)
    data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
    #dataSet['VOL'] = perturb_data(dataSet['VOL'].values,perturbDataLookback)
    #dataSet['OI'] == perturb_data(dataSet['OI'].values,perturbDataLookback)
    
#find max lookback
maxlb = max(RSILookback,
                    zScoreLookback,
                    ATRLookback,
                    DPOLookback,
                    ACLookback)
# add shift
maxlb = maxlb+4

dataSet = data.drop(data.index[data.index.duplicated()], axis=0).copy(deep=True)
nrows = dataSet.shape[0]

#short direction
dataSet['Pri_RSI'] = RSI(dataSet.Close,RSILookback)
dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)

#long direction
dataSet['Pri_DPO'] = DPO(dataSet.Close,DPOLookback)
dataSet['Pri_DPO_Y1'] = dataSet['Pri_DPO'].shift(1)
dataSet['Pri_DPO_Y2'] = dataSet['Pri_DPO'].shift(2)
dataSet['Pri_DPO_Y3'] = dataSet['Pri_DPO'].shift(3)
dataSet['Pri_DPO_Y4'] = dataSet['Pri_DPO'].shift(4)

#volatility
dataSet['Pri_ATR'] = zScore(ATR(dataSet.High,dataSet.Low,dataSet.Close,ATRLookback),
                          zScoreLookback)
dataSet['Pri_ATR_Y1'] = dataSet['Pri_ATR'].shift(1)
dataSet['Pri_ATR_Y2'] = dataSet['Pri_ATR'].shift(2)
dataSet['Pri_ATR_Y3'] = dataSet['Pri_ATR'].shift(3)
dataSet['priceChange'] = priceChange(dataSet.Close)
dataSet['priceChangeY1'] = dataSet['priceChange'].shift(1)
dataSet['priceChangeY2'] = dataSet['priceChange'].shift(2)
dataSet['priceChangeY3'] = dataSet['priceChange'].shift(3)

#correlation
dataSet['autoCor'] = autocorrel(dataSet.Close*100,ACLookback)
dataSet['autoCor_Y1'] = dataSet['autoCor'].shift(1)
dataSet['autoCor_Y2'] = dataSet['autoCor'].shift(2)
dataSet['autoCor_Y3'] = dataSet['autoCor'].shift(3)

#labels
dataSet['gainAhead'] = gainAhead(dataSet.Close)
dataSet['signal'] = np.where(dataSet.gainAhead>beLongThreshold,1,-1)

#mData = dataSet.drop(['Open','High','Low','Close',
#                      'Volume','gainAhead'],
#                      axis=1).dropna()
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


print '\n\nNew simulation run... '
if showDist:
    describeDistribution(dataSet.reset_index()['Close'], dataSet.reset_index()['priceChange'], ticker)

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
print 'Finished Pre-processing... Beginning Model Training..'

#########################################################
model_metrics = init_report()       
sstDictDF1_ = {}

for i,m in enumerate(models):
    for wfStep in wfSteps:
        for wf_is_period in wf_is_periods:
            testFirstYear = dataSet.index[0]
            testFinalYear = dataSet.index[max(wf_is_periods)]
            validationFirstYear =dataSet.index[max(wf_is_periods)+1]
            validationFinalYear =dataSet.index[-1]
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
            runName = ticker+'_'+data_type+'_'+filterName+'_' + m[0]+'_i'+str(wf_is_period)#+'_o'+str(wfStep)
            model_metrics, sstDictDF1_[runName] = wf_regress_validate2(dataSet, dataSet, [m], model_metrics,\
                                                wf_is_period, metaData, PRT, showPDFCDF=showPDFCDF,longMemory=longMemory)
#score models
scored_models, bestModel = directional_scoring(model_metrics,filterName)
if scorePath is not None:
    scored_models.to_csv(scorePath+filterName+'.csv')

#keep original for other DF
#sstDictDF1_Combined_DF1_beShorts_ = copy.deepcopy(sstDictDF1_)
#sstDictDF1_DF1_Shorts_beFlat_ = copy.deepcopy(sstDictDF1_)
if showAllCharts:
    for runName in sstDictDF1_:
        compareEquity(sstDictDF1_[runName],runName)
    
for m in models:
    if bestModel['params'] == str(m[1]):
        print  '\n\nBest model found...\n', m[1]
        bm = m[1]
print 'Number of features: ', bestModel.n_features, bestModel.FS
print 'WF In-Sample Period:', bestModel.rows
print 'WF Out-of-Sample Period:', bestModel.wf_step
print 'Long Memory: ', longMemory
DF1_BMrunName = ticker+'_'+bestModel.data_type+'_'+filterName+'_'  + bestModel.model + '_i'+str(bestModel.rows)
if showAllCharts:
    compareEquity(sstDictDF1_[DF1_BMrunName],DF1_BMrunName)
    
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
print 'Finished Model Training... Beginning Dynamic Position Sizing..'
##############################################################

   
sst_bestModel = sstDictDF1_[DF1_BMrunName].copy(deep=True)

#calc DPS
DPS = {}
DPS_both = {}

startDate = sst_bestModel.index[max(windowLengths)]
endDate = sst_bestModel.index[-1]
for ml in maxLeverage:
    PRT['maxLeverage'] = ml
    for wl in windowLengths:
        #sst_bestModel = pd.concat([pd.Series(data=systems[s].signals, name='signals', index=systems[s].index),\
        #                   systems[s].gainAhead], axis=1)

        #newStart = start            
        #if sst_bestModel.ix[:startDate].shape[0] < wl:
            #zerobegin = pd.DataFrame(data=0, columns= ['signals','gainAhead'],\
            #    index = [sst_bestModel.index[0] - datetime.timedelta(days=x) for x in range(int(math.ceil(wl)),0,-1)])
            #newStart = datetime.datetime.strftime(sst_bestModel.iloc[wl].name,'%Y-%m-%d')
            #print 'both DPS', start,'<=',startDate.date(),'adjusting', start,'to', newStart
            #sst_bestModel = pd.concat([zerobegin, sst_bestModel], axis=0)
            
        dpsRun, sst_save = calcDPS2(DF1_BMrunName, sst_bestModel, PRT, startDate, endDate, wl, 'both', threshold=CAR25_threshold)
        DPS_both[dpsRun] = sst_save
        
        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
        #DPS[dpsRun] = sst_save
    
dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModel, startDate,\
                                            endDate,'both', DF1_BMrunName, yscale='linear',\
                                            ticker=ticker,displayCharts=showAllCharts,\
                                            equityStatsSavePath=equityStatsSavePath)
bestBothDPS.index.name = 'dates'
bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system')], \
                                                axis=1)
print 'Next Signal:'
print bestBothDPS.iloc[-1]
 
print 'Saving Signals..'      
#init file
#bestBothDPS.tail().to_csv(signalPath + version+'_'+ ticker + '.csv')
signal=pd.read_csv(signalPath+ version+'_'+ ticker + '.csv')
signal=signal[:-1].append(bestBothDPS.reset_index().iloc[-2:])
signal.to_csv(signalPath + version+'_'+ ticker + '.csv', index=False)
                
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
