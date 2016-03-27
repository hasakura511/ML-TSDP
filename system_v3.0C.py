
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015
changelog
v3.1
model drop for v1/v2
added wf stepping for ga and zz signals
added verbose mode

v3.0 "KOBE"
added validation period optimization
added more pairs
fixed offline mode
added minute version of CAR25
added timestamp, cycletime

@author: hidemi
"""
import math
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
from pytz import timezone
from datetime import datetime as dt
from os import listdir
from os.path import isfile, join

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
from sklearn.externals import joblib
import arch

#suztoolz
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve,\
                         directional_scoring, compareEquity, describeDistribution,\
                         offlineMode, oos_display_cmatrix
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df_min,\
                            maxCAR25, wf_classify_validate, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS,\
                            calcEquity_df
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
from suztoolz.transform import zigzag as zg
from suztoolz.data import getDataFromIB

start_time = time.time()

#system parameters
version = 'v3'
version_ = 'v3.1'

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='1 min'
currencyPairs = ['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']

                            

def saveModel(dataSet, bestModelParams, verbose=True):
    signal = bestModelParams.signal    
    #perturbDataPct = 0.0002
    #longMemory =  False
    #iterations=1
    #input_signal = 1
    #feature_selection = 'None' #RFECV OR Univariate
    #wfSteps=[1]
    wf_is_period = bestModelParams.rows
    #wf_is_periods = [100]
    #tox_adj_proportion = 0
    #nfeatures = 10

    iterations=10
    RSILookback = bestModelParams.RSILookback
    zScoreLookback = bestModelParams.zScoreLookback
    ATRLookback = bestModelParams.ATRLookback
    beLongThreshold = bestModelParams.beLongThreshold
    DPOLookback = bestModelParams.DPOLookback
    ACLookback = bestModelParams.ACLookback
    CCLookback = bestModelParams.CCLookback
    rStochLookback = bestModelParams.rStochLookback
    statsLookback = bestModelParams.statsLookback
    ROCLookback = bestModelParams.ROCLookback
    model = eval(bestModelParams.params)
    
    mData = dataSet.drop(['Open','High','Low','Close',
                           'Volume','prior_index','gainAhead'],
                            axis=1).dropna()    
    #  Select the date range to test no label for the last index
    mmData = mData.iloc[-wf_is_period:-1]

    datay = mmData.signal
    mmData = mmData.drop(['signal'],axis=1)

    print '\nIn-Sample Period: %i ' % wf_is_period
    print 'Total %i features: ' % mmData.shape[1]
    for i,x in enumerate(mmData.columns):
        print i,x+',',
        #feature_names = feature_names+[x]    
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


    
    if verbose:
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
    return model

def wf_classify_validate2(unfilteredData, dataSet, models, model_metrics, \
                           metaData, **kwargs):
    showPDFCDF= kwargs.get('showPDFCDF',True)
    showLearningCurve=kwargs.get('showLearningCurve',False)
    longMemory=kwargs.get('longMemory',False)
    verbose=kwargs.get('verbose',True)
    
    close = unfilteredData.reset_index().Close
    #fill in the prior index. need this for the car25 calc uses the close index
    unfilteredData['prior_index'] = pd.concat([dataSet.prior_index, unfilteredData.Close],axis=1,join='outer').prior_index.interpolate(method='linear').dropna()
    ticker = metaData['ticker']
    data_type = metaData['data_type']
    iterations = metaData['iters']
    testFinalYear= metaData['t_end']
    validationFirstYear=metaData['v_start']
    validationFinalYear=metaData['v_end']
    wfStep=metaData['wf_step']
    signal =  metaData['signal']
    nfeatures = metaData['n_features']
    tox_adj_proportion = metaData['tox_adj']
    feature_selection = metaData['FS']
    wf_is_period = metaData['wf_is_period']
    ddTolerance = metaData['DD95_limit']
    forecastHorizon = metaData['horizon']
    
    #CAR25 for zigzag
    if signal[:2]=='ZZ':
        zz_step = [float(x) for x in signal.split('_')[1].split(',')]

    #create signals
    if signal != 'GA1' or signal != 'gainAhead':
        if signal[:2] == 'GA':
            #wfStep = int(signal[2:])
            #ga_start = mmData_v.iloc[train_index].prior_index.iloc[0]
            ga_start = dataSet.iloc[0].prior_index
            ga = gainAhead(close.ix[ga_start:],wfStep)
            dataSet.signal = np.array([-1 if x<0 else 1 for x in ga])
            
        if signal[:2] == 'ZZ':
            zz_end = dataSet.iloc[-1].prior_index
            dataSet.signal = pd.Series(zg(close.ix[:zz_end].values, zz_step[0], \
                            zz_step[1]).pivots_to_modes()[-dataSet.shape[0]:]).shift(-1).fillna(0).values
    metaData['wf_step']=wfStep


    if 'filter' in metaData:
        filterName = metaData['filter']
    else:
        filterName = 'OOS_V'

    dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']

    #check
    nrows_is = dataSet.ix[:testFinalYear].dropna().shape[0]
    if wf_is_period > nrows_is:
        print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
        print 'Adjusting to', nrows_is, 'rows..'
        wf_is_period = nrows_is

    mmData = dataSet.ix[:testFinalYear].dropna()[-wf_is_period:]
    mmData_adj = adjustDataProportion(mmData, tox_adj_proportion)  #drop last row for hold days =1
    mmData_v = pd.concat([mmData_adj,dataSet.ix[validationFirstYear:validationFinalYear].dropna()], axis=0).reset_index()

    nrows_is = mmData.shape[0]
    nrows_oos = mmData_v.shape[0]-nrows_is
        
    metaData['rows'] = nrows_is

    #nrows = mmData_adj.shape[0]
    datay_signal = mmData_v[['signal', 'prior_index']]
    datay_gainAhead = mmData_v.gainAhead

    dataX = mmData_v.drop(dropCol, axis=1) 
    cols = dataX.columns.shape[0]
    metaData['cols']=cols
    
    feature_names = []
    if verbose == True:
        print '\nTotal %i features: ' % cols
    for i,x in enumerate(dataX.columns):
        if verbose == True:
            print i,x+',',
        feature_names = feature_names+[x]
        
    if feature_selection is not 'None':
        if nfeatures > cols:
            print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
            print 'Adjusting to', cols, 'features..'
            nfeatures = cols  
        metaData['cols']=nfeatures
    
            
    #  Copy from pandas dataframe to numpy arrays
    dy = np.zeros_like(datay_signal.signal)
    dX = np.zeros_like(dataX)

    dy = datay_signal.signal.values
    dX = dataX.values
    for m in models:
        if verbose == True:
            print '\n\nNew WF train/predict loop for', m[1]
            print "\nStarting Walk Forward run on", metaData['data_type'], "data..."
            if feature_selection == 'Univariate':
                print "Using top %i %s features" % (nfeatures, feature_selection)
            else:
                print "Using features selection: %s " % feature_selection
            if longMemory == False:
                print "%i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (nrows_is, nrows_oos,wfStep)
            else:
                print "long memory starting with %i rows in sample, %i rows out of sample, forecasting %i bar(s) ahead.." % (nrows_is, nrows_oos,wfStep)
            #cm_y_train = np.array([])
        cm_y_test = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        leftoverIndex = nrows_oos%wfStep
        
        #reverse index to equate the wf tests of different periods, count backwards from the end
        wfIndex = range(nrows_oos-wfStep,-wfStep,-wfStep)
        tt_index =[]
        for i in wfIndex:
            #last wf index adjust the test index, else step
            if leftoverIndex > 0 and i == wfIndex[-1]:
                train_index = range(0,wf_is_period)        
                test_index = range(wf_is_period,wf_is_period+leftoverIndex)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
            else:
                if longMemory == True:
                    train_index = range(0,wf_is_period+i)
                else:
                    train_index = range(i,wf_is_period+i)
                #the last wfStep indexes are untrained.
                test_index = range(wf_is_period+i,wf_is_period+i+wfStep)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
        #c=0
        #zz_begin = mmData_v.prior_index.iloc[0]
        for train_index,test_index in tt_index:
            #c+=1
            X_train, X_test = dX[train_index], dX[test_index]
            y_train, y_test = dy[train_index], dy[test_index]
            
            #create zigzag signals
            
            #ending at test_index so dont need to shift labels
            #if signal[:3] != 'GA1':
            #    if signal[:2] == 'GA':
            #        lookforward = int(signal_types[2][2:])
            #        ga_start = mmData_v.iloc[train_index].prior_index.iloc[0]
            #        #ga_start = dataSet.iloc[0].prior_index
            #        ga = gainAhead(close.ix[ga_start:],lookforward)
            #        y_train = np.array([-1 if x<0 else 1 for x in ga])[:len(train_index)]
                    
            if signal[:2] == 'ZZ':
                zz_end = mmData_v.iloc[test_index].prior_index.iloc[len(test_index)-1]
                y_train = zg(close.ix[:zz_end].values, zz_step[0], \
                                zz_step[1]).pivots_to_modes()[-len(train_index):]
                
            #check if there are no intersections
            intersect = np.intersect1d(datay_signal.reset_index().iloc[test_index]['index'].values,\
                        datay_signal.reset_index().iloc[train_index]['index'].values)
            if intersect.size != 0:
                print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
            if len(mmData_v.index[-wfStep:].intersection(train_index)) == 0:
                #print 'training', X_train.shape
                if feature_selection is not 'None':
                    if feature_selection == 'RFECV':
                        #Recursive feature elimination with cross-validation: 
                        #A recursive feature elimination example with automatic tuning of the
                        #number of features selected with cross-validation.
                        rfe = RFECV(estimator=RFE_estimator, step=1)
                        rfe.fit(X_train, y_train)
                        #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
                        featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
                        print 'Top %i RFECV features' % len(featureRank)
                        print featureRank    
                        metaData['featureRank'] = str(featureRank)
                        X_train = rfe.transform(X_train)
                        X_test = rfe.transform(X_test)
                    else:
                        #Univariate feature selection
                        skb = SelectKBest(f_regression, k=nfeatures)
                        skb.fit(X_train, y_train)
                        #dX_all = np.vstack((X_train.values, X_test.values))
                        #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
                        #dX_v_rfe = X_new[dX_t.shape[0]:]
                        X_train = skb.transform(X_train)
                        X_test = skb.transform(X_test)
                        featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
                        metaData['featureRank'] = str(featureRank)
                        #print 'Top %i univariate features' % len(featureRank)
                        #print featureRank

                #  fit the model to the in-sample data
                m[1].fit(X_train, y_train)
                print X_train.shape

                #trained_models[m[0]] = pickle.dumps(m[1])
                            
                #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))              
                y_pred_oos = m[1].predict(X_test)

                if m[0][:2] == 'GA':
                    print featureRank
                    print '\nProgram:', m[1]._program
                    #print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
                
                #cm_y_train = np.concatenate([cm_y_train,y_train])
                cm_y_test = np.concatenate([cm_y_test,y_test])
                #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
                cm_y_pred_oos = np.concatenate([cm_y_pred_oos,y_pred_oos])
                #cm_train_index = np.concatenate([cm_train_index,train_index])
                cm_test_index = np.concatenate([cm_test_index,test_index])
            

        #create signals 1 and -1
        #cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos_ga])
        #cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test_ga])
        
        #gives errors when 100% accuracy for binary classification
        #if confusion_matrix(cm_y_test[:-1], cm_y_pred_oos[:-1]).shape == (1,1):
        #    print  m[0], ticker,validationFirstYear, validationFinalYear, iterations, signal
        #    print 'Accuracy 100% for', cm_y_test[:-1].shape[0], 'rows'
        #else:
        if verbose == True:
            if wfStep>1:
                oos_display_cmatrix(cm_y_test[:-wfStep], cm_y_pred_oos[:-wfStep], m[0],\
                    ticker,validationFirstYear, dataSet.index[-wfStep], iterations, signal)
            else:
                oos_display_cmatrix(cm_y_test[:-1], cm_y_pred_oos[:-1], m[0],\
                        ticker,validationFirstYear, validationFinalYear, iterations, signal)
        #if data is filtered so need to fill in the holes. signal = 0 for days that filtered
        st_oos_filt= pd.DataFrame()
        st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
        st_oos_filt.index = mmData_v['dates'].iloc[cm_test_index]
                
        #compute car, show matrix if data is filtered
        if data_type != 'ALL':
            
            prior_index_filt = pd.concat([st_oos_filt,unfilteredData.prior_index], axis=1,\
                                join='inner').prior_index.values.astype(int)
            #datay_gainAhead and cm_test_index have the same index. dont need to have same shape because iloc is used in display
            if verbose == True:
                print 'Metrics for filtered Validation Datapoints'
                oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                        ticker, validationFirstYear, validationFinalYear, iterations, metaData['filter'],showPDFCDF)
            CAR25_oos = CAR25_df_min(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, minFcst=forecastHorizon, DD95_limit =ddTolerance, verbose=verbose)
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'LONG', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'SHORT', -1)
                                    
        #add column prior index and gA.  if there are holes, nan values in signals
        st_oos_filt = pd.concat([st_oos_filt,unfilteredData.gainAhead,unfilteredData.prior_index],\
                                    axis=1, join='outer').ix[validationFirstYear:validationFinalYear]
        #fills nan with zeros
        st_oos_filt['signals'].fillna(0, inplace=True)
        
        #fill zeros with opposite of input signal, if there are zeros. to return full data
        cm_y_pred_oos = np.where(st_oos_filt['signals'].values==0,metaData['input_signal']*-1,\
                                                                    st_oos_filt['signals'].values)
        cm_y_test = np.where(st_oos_filt.gainAhead>0,1,-1)
        #datay_gainAhead and cmatrix_test_index have the same index
        datay_gainAhead = st_oos_filt.gainAhead
        cmatrix_test_index = st_oos_filt.reset_index().index

        #plot learning curve, knn insufficient neighbors
        if showLearningCurve:
            try:
                plot_learning_curve(m[1], m[0], X_train,y_train_ga, scoring='r2')        
            except:
                pass

        
        #compute car, show matrix for all data is unfiltered
        if data_type == 'ALL':           
            if verbose == True:
                print 'Metrics for All Validation Datapoints'
                oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cmatrix_test_index, m[1], ticker,\
                                    validationFirstYear, validationFinalYear, iterations, 'Long>0',showPDFCDF)
            CAR25_oos = CAR25_df_min(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, minFcst=forecastHorizon, DD95_limit =ddTolerance, verbose=verbose)
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
             #                       close, 'LONG', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
             #                       close, 'SHORT', -1)
        #update model metrics
        #metaData['signal'] = 'LONG 1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
                                cmatrix_test_index, m, metaData,CAR25_oos)
        #metaData['signal'] = 'SHORT -1'
        #model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
        #                       cmatrix_test_index, m, metaData,CAR25_Sn1_oos)
    return model_metrics, st_oos_filt, m[1]
        
#no args -> debug.  else live mode arg 1 = pair, arg 2 = "0" to turn off
if len(sys.argv)==1:
    debug=True
    
    livePairs =  [
                    #'NZDJPY',\
                    #'CADJPY',\
                    #'CHFJPY',\
                    #'EURJPY',\
                    #'GBPJPY',\
                    #'AUDJPY',\
                    #'USDJPY',\
                    'AUDUSD',\
                    #'EURUSD',\
                    #'GBPUSD',\
                    #'USDCAD',\
                    #'USDCHF',\
                    #'NZDUSD',
                    #'EURCHF',\
                    #'EURGBP'\
                    ]
                    
                    
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = True
    runDPS = True
    verbose= False
    #scorePath = './debug/scored_metrics_'
    #equityStatsSavePath = './debug/'
    #signalPath = './debug/'
    #dataPath = './data/from_IB/'
    scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    modelSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/models/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    
else:
    print 'Live Mode', sys.argv[1], sys.argv[2]
    debug=False
    
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    runDPS = True
    verbose= False
    
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    bestParamsPath =  './data/params/'
    modelSavePath = './data/models/' 
    
    if sys.argv[2] == "0":
        livePairs=[]
        ticker = sys.argv[1]
        offlineMode(ticker, "Offline Mode: "+sys.argv[0]+' '+sys.argv[1], signalPath, version, version_)
    else:
        livePairs=[sys.argv[1]]
           

        
for ticker in livePairs:
    print 'Begin optimization run for', ticker
    symbol=ticker[0:3]
    currency=ticker[3:6]

    #Model Parameters
    #dataSet length needs to be divisiable by each validation period! 
    validationSetLength = 480
    #validationSetLength = 1200
    #validationPeriods = [50,250]
    validationPeriods = [120,240] # min is 2
    #validationStartPoint = None
    #signal_types = ['gainAhead','ZZ']
    signal_types = ['ZZ']
    #signal_types = ['gainAhead']
    zz_steps = [0.006,0.009]
    #zz_steps = [0.009]
    wfSteps=[60]
    wf_is_periods = [250,500]
    #wf_is_periods = [250,500,1000]
    perturbDataPct = 0.0002
    longMemory =  False
    iterations=1
    input_signal = 1
    feature_selection = 'None' # OR Univariate
    #feature_selection = 'RFECV'
    #feature_selection = 'Univariate'
    nfeatures = 10
    tox_adj_proportion = 0
    

    #feature Parameters
    RSILookback = 1.5
    zScoreLookback = 10
    ATRLookback = 5
    beLongThreshold = 0.0
    DPOLookback = 10
    ACLookback = 12
    CCLookback = 60
    rStochLookback = 300
    statsLookback = 100
    ROCLookback = 600

    #DPS parameters
    #windowLengths = [2] 
    windowLengths = [30]
    maxLeverage = [5]
    PRT={}
    PRT['DD95_limit'] = 0.01
    PRT['tailRiskPct'] = 95
    PRT['initial_equity'] = 1.0
    PRT['horizon'] = 720
    PRT['maxLeverage'] = 5
    #CAR25_threshold=-np.inf
    CAR25_threshold=0

    #model selection
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    RFE_estimator = [ 
            ("None","None"),\
            #("GradientBoostingRegressor",GradientBoostingRegressor()),\
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
              #                             n_jobs=1, verbose=0, random_state=None)),
             #("GA_Reg2", SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, comparison=True, transformer=False, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=1, verbose=0, parsimony_coefficient=0.01, random_state=0)),
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             #("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:1}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             ("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 ("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                    ], voting='hard', weights=None)),
             #("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #        ], voting='soft', weights=None)),
             ]    


    ############################################################
    currencyPairsDict = {}
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

    for pair in currencyPairs:    
        if barSizeSetting+'_'+pair+'.csv' in files:
            data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)
            currencyPairsDict[pair] = data
            
    print 'Successfully Retrieved Data. Beginning Preprocessing...'
    ###########################################################
    #print 'Begin Preprocessing...'

    #add cross pair features
    #print 'removing dupes..'
    currencyPairsDict2 = copy.deepcopy(currencyPairsDict)
    for pair in currencyPairsDict2:
        data = currencyPairsDict2[pair].copy(deep=True)
        #perturb dataSet
        if perturbData:
            if verbose:
                print 'perturbing OHLC and dropping dupes',
            data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
            data['High']= perturb_data(data['High'].values,perturbDataPct)
            data['Low']= perturb_data(data['Low'].values,perturbDataPct)
            data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
            
        if verbose:
            print pair, currencyPairsDict2[pair].shape,'to',
        currencyPairsDict2[pair] = data.drop_duplicates()
        if verbose:
            print pair, currencyPairsDict2[pair].shape

    #if last index exists in all pairs append to dataSet
    dataSet = currencyPairsDict2[ticker].copy(deep=True)

    for pair in currencyPairsDict2:
        if dataSet.shape[0] != currencyPairsDict2[pair].shape[0]:
            if verbose:
                print 'Warning:',ticker, pair, 'row mismatch. Some Data may be lost.'
            #print 'Adjusting rows in dataSet and cross-pair to match', ticker, dataSet.shape[0],\
            #            pair, currencyPairsDict2[pair].shape[0], 'to',
            #intersect =np.intersect1d(currencyPairsDict2[pair].index, dataSet.index)         
            #dataSet = dataSet.ix[intersect]:
            #currencyPairsDict2[pair] = currencyPairsDict2[pair].ix[intersect]
            #print ticker, dataSet.shape[0], pair, currencyPairsDict2[pair].shape[0]
                            
    nrows = dataSet.shape[0]
    lastIndex = currencyPairsDict2[ticker].index[-1]
    for pair in currencyPairsDict2:
        if lastIndex in currencyPairsDict2[pair].index and pair != ticker and\
                            (ticker[0:3] in pair or ticker[3:6] in pair):
            intersect =np.intersect1d(currencyPairsDict2[pair].index, dataSet.index)         
            dataSet = dataSet.ix[intersect]
            currencyPairsDict2[pair] = currencyPairsDict2[pair].ix[intersect]
            
            closes = pd.concat([dataSet.Close, currencyPairsDict2[pair].Close],\
                                    axis=1, join='inner')
                                                           
            dataSet['corr'+pair] = pd.rolling_corr(closes.iloc[:,0],\
                                            closes.iloc[:,1], window=CCLookback)
            dataSet['priceChange'+pair] = priceChange(closes.iloc[:,1])
            dataSet['Pri_rStoch'+pair] = roofingFilter(closes.iloc[:,1],rStochLookback,500)
            dataSet['ROC_'+pair] = ROC(closes.iloc[:,1],ROCLookback)
    
    print nrows-dataSet.shape[0], 'rows lost for', ticker
    
    #short indicators
    dataSet['Pri_RSI'] = RSI(dataSet.Close,RSILookback)
    dataSet['Pri_RSI_Y1'] = dataSet['Pri_RSI'].shift(1)
    dataSet['Pri_RSI_Y2'] = dataSet['Pri_RSI'].shift(2)
    dataSet['Pri_RSI_Y3'] = dataSet['Pri_RSI'].shift(3)
    dataSet['Pri_RSI_Y4'] = dataSet['Pri_RSI'].shift(4)

    #long indicators
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
    dataSet['signal'] =  np.where(dataSet.gainAhead>0,1,-1)
    
    for wfStep in wfSteps:
        print '\nCreating Signal labels..'
        if 'ZZ' in signal_types:
            signal_types.remove('ZZ')
            for i in zz_steps:
                #for j in zz_steps:
                label = 'ZZ'+str(wfStep)+'_'+str(i) + ',-' + str(i)
                print label+',',
                signal_types.append(label)
                #zz_signals[label] = zg(dataSet.Close, i, -j).pivots_to_modes()
                
        if 'gainAhead' in signal_types:
            signal_types.remove('gainAhead')
            label = 'GA'+str(wfStep)
            print label+',',
            signal_types.append(label)
            
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
    unfilteredData = dataSet.copy(deep=True)
    
    if dataSet.shape[0] <validationSetLength+max(wf_is_periods):
        message = 'Add more data: dataSet rows '+str(dataSet.shape[0])+\
                    ' is less than required validation set + training set of '\
                    + str(validationSetLength+max(wf_is_periods))
        offlineMode(ticker, message, signalPath, version, version_)
    else:
        for i in validationPeriods:
            if dataSet.iloc[-validationSetLength:].shape[0]%i != 0:
                message='validationSetLength '+str(validationSetLength)+\
                        ' needs to be divisible by validation period '+ str(i)
                offlineMode(ticker, message, signalPath, version, version_)
            if validationSetLength == i:
                message='validationSetLength '+str(validationSetLength)+\
                        ' needs to be less than validation period '+ str(i)
                offlineMode(ticker, message, signalPath, version, version_)
                
        dataSet = dataSet.iloc[-validationSetLength-max(wf_is_periods):]
        print '\nProcessed ', dataSet.shape[0], 'rows from', dataSet.index[0], 'to', dataSet.index[-1]

        
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    
    if showDist:
        describeDistribution(dataSet.reset_index()['Close'], dataSet.reset_index()['priceChange'], ticker)  
    #print 'Finished Pre-processing... Beginning Model Training..'

    #########################################################
    validationDict = {}
    BMdict = {}
    BSMdict={}
    for validationPeriod in validationPeriods:
        #DPScycle = 0
        endOfData = 0
        t_start_loc = max(wf_is_periods)
        if validationPeriod > dataSet.shape[0]+t_start_loc:
            print 'Validation Period', validationPeriod, '> dataSet + training period',
            validationPeriod = dataSet.shape[0]-t_start_loc-1
            print 'adjusting to', validationPeriod, 'rows'
        elif validationPeriod <2:
            print 'Validation Period', validationPeriod, '< 2',
            validationPeriod = 2
            print 'adjusting to 2 rows'
        
        testFirstYear = dataSet.index[0]
        testFinalYear = dataSet.index[t_start_loc-1]
        validationFirstYear =dataSet.index[t_start_loc]   
        validationFinalYear =dataSet.index[t_start_loc+validationPeriod-1]
        
        #check if validation start point is less than available training data
        #if validationStartPoint is not None and\
        #            dataSet.shape[0]>dataSet.index[-validationStartPoint:].shape[0]+t_start_loc:
        #    testFinalYear = dataSet.index[-validationStartPoint-1]
        #    validationFirstYear =dataSet.index[-validationStartPoint]
            
        while endOfData<2:
            print '\nStarting new simulation run from', validationFirstYear, 'to', validationFinalYear
            #init
            model_metrics = init_report()       
            sstDictDF1_ = {} 
            SMdict={}
            #DPScycle+=1
            
            for signal in signal_types:
                for m in models:
                    #for wfStep in wfSteps:
                    wfStep=int(signal.split('_')[0][2:])
                    for wf_is_period in wf_is_periods:

                        #check
                        nrows_is = dataSet.ix[:testFinalYear].shape[0]
                        if wf_is_period > nrows_is:
                            print 'Walkforward insample period of', wf_is_period,\
                                    'is greater than in-sample data of ', nrows_is, '!'
                            print 'Adjusting to', nrows_is, 'rows..'
                            wf_is_period = nrows_is
  
                        metaData = {'ticker':ticker, 't_start':testFirstYear, 't_end':testFinalYear,\
                                'signal':signal, 'data_type':data_type,'filter':filterName, 'input_signal':input_signal,\
                                'test_split':0, 'iters':1, 'tox_adj':tox_adj_proportion,'longMemory':longMemory,\
                                'n_features':nfeatures, 'FS':feature_selection,'rfe_model':RFE_estimator[0],\
                                'v_start':validationFirstYear, 'v_end':validationFinalYear,'wf_step':wfStep,\
                                'RSILookback': RSILookback,'zScoreLookback': zScoreLookback,'ATRLookback': ATRLookback,\
                                'beLongThreshold': beLongThreshold,'DPOLookback': DPOLookback,'ACLookback': ACLookback,\
                                'CCLookback': CCLookback,'rStochLookback': rStochLookback,'statsLookback': statsLookback,\
                                'ROCLookback': ROCLookback, 'DD95_limit':PRT['DD95_limit'],'tailRiskPct':PRT['tailRiskPct'],\
                                'initial_equity':PRT['initial_equity'],'horizon':PRT['horizon'],'maxLeverage':PRT['maxLeverage'],\
                                'wf_is_period':wf_is_period
                                 }
                        runName = ticker+'_'+data_type+'_'+filterName+'_' + m[0]+'_i'+str(wf_is_period)+'_fcst'+str(wfStep)+'_'+signal
                        model_metrics, sstDictDF1_[runName], SMdict[runName] = wf_classify_validate2(unfilteredData, dataSet, [m], model_metrics,\
                                                                                                                    metaData, showPDFCDF=showPDFCDF,\
                                                                                                                    verbose=verbose)

            #score models
            scored_models, bestModelParams = directional_scoring(model_metrics,filterName)
            #bestModelParams['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            #bestModelParams = bestModelParams.append(pd.Series(data=datetime.datetime.fromtimestamp(time.time())\
            #                               .strftime('%Y-%m-%d %H:%M:%S'), index=['timestamp']))
                                            
            #keep original for other DF
            #sstDictDF1_Combined_DF1_beShorts_ = copy.deepcopy(sstDictDF1_)
            #sstDictDF1_DF1_Shorts_beFlat_ = copy.deepcopy(sstDictDF1_)
            if showAllCharts:
                for runName in sstDictDF1_:
                    compareEquity(sstDictDF1_[runName],runName)
                
            for m in models:
                if bestModelParams['params'] == str(m[1]):
                    print  '\nBest model found...\n', m[1]
                    bm = m[1]
            print 'Feature selection: ', bestModelParams.FS
            print 'Number of features: ', bestModelParams.cols
            print 'WF In-Sample Period:', bestModelParams.rows
            print 'WF Out-of-Sample Period:', bestModelParams.wf_step
            print 'Long Memory: ', longMemory
            DF1_BMrunName = ticker+'_'+bestModelParams.data_type+'_'+filterName+'_'  +\
                                bestModelParams.model + '_i'+str(bestModelParams.rows)\
                                +'_fcst'+str(bestModelParams.wf_step)+'_'+bestModelParams.signal
                
            if showAllCharts:
                compareEquity(sstDictDF1_[DF1_BMrunName],DF1_BMrunName)
                
            #save best model file on last iteration
            #if endOfData ==1:
            #   modelToSave = saveModel(dataSet, bestModelParams)
            
            print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
            #print 'Finished Model Training...'
            
            if runDPS:               
                print '\nDynamic Position Sizing..'
                for wl in windowLengths:
                    if validationPeriod in validationDict and \
                           len([x for x in validationDict[validationPeriod].index if x not in sstDictDF1_[DF1_BMrunName].index]) >wl:
                    #prior validation data available from the third DPScycle. wl<1st cycle vp
                        sst_bestModelParams = validationDict[validationPeriod].drop(['safef','CAR25','dd95','ddTol','system'], axis=1)
                        zero_index = np.array([x for x in sst_bestModelParams.index if x not in sstDictDF1_[DF1_BMrunName].index])[-int(wl):]
                        sst_zero = sst_bestModelParams.ix[zero_index]
                        sst_bestModelParams = pd.concat([sst_zero,sstDictDF1_[DF1_BMrunName]],axis=0)
                        
                        #adjust window length if it is larger than zero index
                        if sst_zero.shape[0]<wl:
                            wl=sst_zero.shape[0]
                    else:
                        #add zeros if two cycles
                        zero_index = dataSet.ix[:sstDictDF1_[DF1_BMrunName].index[0]].index[:-1]
                        sst_zero = pd.concat([pd.Series(data=0, name='signals', index=zero_index ), dataSet.gainAhead.ix[zero_index],\
                                                        dataSet.prior_index.ix[zero_index]], axis=1)
                        sst_bestModelParams = pd.concat([sst_zero,sstDictDF1_[DF1_BMrunName]],axis=0)
                        
                        #adjust window length if it is larger than zero index
                        if sst_zero.shape[0]<wl:
                            wl=sst_zero.shape[0]

                    #calc DPS
                    #DPS = {}
                    DPS_both = {}

                    startDate = sstDictDF1_[DF1_BMrunName].index[0]
                    endDate = sst_bestModelParams.index[-1]
                    for ml in maxLeverage:
                        PRT['maxLeverage'] = ml               
                        dpsRun, sst_save = calcDPS2(DF1_BMrunName, sst_bestModelParams, PRT, startDate,\
                                                                    endDate, wl, 'both', threshold=CAR25_threshold)
                        DPS_both[dpsRun] = sst_save
                        
                        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                        #DPS[dpsRun] = sst_save
                    
                dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModelParams, startDate,\
                                                            endDate,'both', DF1_BMrunName, yscale='linear',\
                                                            ticker=ticker,displayCharts=showAllCharts,\
                                                            equityStatsSavePath=equityStatsSavePath, verbose=verbose)
                bestBothDPS.index.name = 'dates'
                bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system')], \
                                                                axis=1)
                if endOfData ==1:
                    #save sst's for save file         
                    BMdict[validationPeriod] = bestBothDPS
                    #save model
                    BSMdict[validationPeriod] = SMdict[DF1_BMrunName]
                    
                if showAllCharts:
                    DPSequity = calcEquity_df(bestBothDPS[['signals','gainAhead']], dpsRunName, leverage = bestBothDPS.safef.values)
            elif endOfData ==1:
                BMdict[validationPeriod] = pd.concat([sstDictDF1_[DF1_BMrunName],\
                                    pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[DF1_BMrunName].index),
                                    pd.Series(data=DF1_BMrunName, name = 'system', index = sstDictDF1_[DF1_BMrunName].index)
                                    ],axis=1)
                #save model
                BSMdict[validationPeriod] = SMdict[DF1_BMrunName]
                
            #use best params for next step
            for i in range(0,bestModelParams.shape[0]):
                metaData[bestModelParams.index[i]]=bestModelParams[bestModelParams.index[i]]
             
            #adjust test dates
            t_start_loc = dataSet.reset_index()[dataSet.reset_index().dates ==bestModelParams.v_end].index[0]
            testFirstYear = dataSet.index[0]
            testFinalYear = dataSet.index[t_start_loc]
            
            #if t_start_loc+1 >= dataSet.shape[0]:
            #    validationFirstYear =dataSet.index[t_start_loc]
            #else:
            if endOfData<1:
                validationFirstYear = dataSet.index[t_start_loc+1]
                validationFinalYear =dataSet.index[t_start_loc+validationPeriod]
            
            #if t_start_loc+validationPeriod >= dataSet.shape[0]:
            if validationFinalYear == dataSet.index[-1]:
                endOfData+=1
            #else:
            
            
            
            if endOfData<2:
                print '\nSimulating Next Period from', validationFirstYear, 'to',validationFinalYear
                #update metaData dates
                metaData['t_start']=testFirstYear
                metaData['t_end']=testFinalYear
                metaData['v_start']=validationFirstYear
                metaData['v_end']=validationFinalYear
                nextRunName = 'Next'+str(validationPeriod)+'_'+DF1_BMrunName
                model_metrics, sstDictDF1_[nextRunName], savedModel = wf_classify_validate2(unfilteredData, dataSet, [m], model_metrics,\
                                                    metaData, showPDFCDF=showPDFCDF, verbose=verbose)
                if showAllCharts:
                    compareEquity(sstDictDF1_[nextRunName],nextRunName)
                print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
                #print 'Finished Next Period Model Training... '   
                if runDPS:
                    #if DPS was chosen as the best model to use going forward
                    if dpsRunName.find('wl') != -1:
                        #for ml in maxLeverage:
                        PRT['maxLeverage'] = float(dpsRunName[dpsRunName.find('wl'):].split()[1][4:])
                        #for wl in windowLengths:
                        wl=float(dpsRunName[dpsRunName.find('wl'):].split()[0][2:])
                        
                        #print 'Beginning Next Period Dynamic Position Sizing..'
                        if validationPeriod in validationDict and validationDict[validationPeriod].shape>=wl:
                            sst_bestModelParams = pd.concat([validationDict[validationPeriod].drop(['safef','CAR25','dd95','ddTol','system'],axis=1),\
                                                                sstDictDF1_[nextRunName]], axis=0)
                        else:
                            #sst_bestModelParams = pd.concat([sstDictDF1_[DF1_BMrunName],\
                            #                                   sstDictDF1_[nextRunName]], axis=0)
                                                                
                            #add zeros if two cycles
                            zero_index = np.array([x for x in sstDictDF1_[DF1_BMrunName].index\
                                                                if x not in sstDictDF1_[nextRunName].index])[-int(wl):]
                            sst_zero = pd.concat([pd.Series(data=0, name='signals', index=zero_index ),\
                                                            sstDictDF1_[DF1_BMrunName].gainAhead.ix[zero_index],\
                                                            sstDictDF1_[DF1_BMrunName].prior_index.ix[zero_index]], axis=1)
                            sst_bestModelParams = pd.concat([sst_zero,sstDictDF1_[nextRunName]],axis=0)
                            
                            #adjust window length if it is larger than zero index
                            if sst_zero.shape[0]<wl:
                                wl=sst_zero.shape[0]
                            
                        #calc DPS
                        DPS_both = {}

                        startDate = sstDictDF1_[nextRunName].index[0]
                        endDate = sst_bestModelParams.index[-1]
                        dpsRun, sst_save = calcDPS2(nextRunName, sst_bestModelParams, PRT, startDate,\
                                                                    endDate, wl, 'both', threshold=CAR25_threshold)
                        DPS_both[dpsRun] = sst_save
                        
                        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
                        #DPS[dpsRun] = sst_save
                            
                        dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, sst_bestModelParams, startDate,\
                                                                    endDate,'both', nextRunName, yscale='linear',\
                                                                    ticker=ticker,displayCharts=showAllCharts,\
                                                                    equityStatsSavePath=equityStatsSavePath, verbose=verbose)
                        bestBothDPS.index.name = 'dates'
                        bestBothDPS = pd.concat([bestBothDPS, pd.Series(data = dpsRunName,index=bestBothDPS.index, name='system'),\
                                                                ], axis=1)
                        if showAllCharts:                                       
                            DPSequity = calcEquity_df(bestBothDPS[['signals','gainAhead']], dpsRunName, leverage = bestBothDPS.safef.values)
                        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
                        #print 'Finished Dynamic Position Sizing..'
                        
                        if validationPeriod not in validationDict:
                            validationDict[validationPeriod] =bestBothDPS
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                    message = "Error: NaN in bestBothDPS.prior_index:1 "
                                    offlineMode(ticker, message, signalPath, version, version_)
                        else:              
                            validationDict[validationPeriod] =validationDict[validationPeriod].append(bestBothDPS)
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                    message="Error: NaN in bestBothDPS.prior_index:2 "
                                    offlineMode(ticker, message, signalPath, version, version_)
                    else:
                        #if No DPS was chosen as the best model to use going forward
                        if validationPeriod not in validationDict:
                            validationDict[validationPeriod] =    pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                message="Error: NaN in bestBothDPS.prior_index:3 "
                                offlineMode(ticker, message, signalPath, version, version_)    
                        else:                      
                            validationDict[validationPeriod] =validationDict[validationPeriod].append(                                                                         
                                                                              pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)                                                               
                                                                             )
                            if sum(np.isnan(validationDict[validationPeriod].prior_index.values))>0:
                                message="Error: NaN in bestBothDPS.prior_index:4"
                                offlineMode(ticker, message, signalPath, version, version_)    
                               
                else:   
                    #DPSrun == False
                    if validationPeriod not in validationDict:
                        validationDict[validationPeriod] =pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)
                    else:
                        validationDict[validationPeriod] =validationDict[validationPeriod].append(                                                                         
                                                                              pd.concat([sstDictDF1_[nextRunName],\
                                                                            pd.Series(data=1.0, name = 'safef', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'CAR25', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'dd95', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=np.nan, name = 'ddTol', index = sstDictDF1_[nextRunName].index),
                                                                            pd.Series(data=nextRunName, name = 'system', index = sstDictDF1_[nextRunName].index)
                                                                            ],axis=1)                                                               
                                                                             )
          
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print '\n\nScoring Validation Curves...'
    #set validation curves equal length. end date is -2 because GA at [-1] is 0.
    vStartDate = dataSet.index[0]
    vEndDate = dataSet.index[-2]
    for validationPeriod in validationDict:
        if validationDict[validationPeriod].index[0]>vStartDate:
            vStartDate =  validationDict[validationPeriod].index[0]
    
    if runDPS:
        vCurve_metrics = init_report()
        for validationPeriod in validationDict:
            vCurve = ticker+' Validation Period '+str(validationPeriod)
            #compareEquity(validationDict[validationPeriod],vCurve)
            if showAllCharts:
                DPSequity = calcEquity_df(validationDict[validationPeriod].ix[vStartDate:vEndDate][['signals','gainAhead']], vCurve,\
                                                    leverage = validationDict[validationPeriod].ix[vStartDate:vEndDate].safef.values)
            if scorePath is not None:
                validationDict[validationPeriod].to_csv(equityStatsSavePath+vCurve+'_'+\
                                                                str(validationDict[validationPeriod].index[0]).replace(':','')+'_to_'+\
                                                                str(validationDict[validationPeriod].index[-1]).replace(':','')+'.csv')
            CAR25_oos = CAR25_df_min(vCurve,validationDict[validationPeriod].ix[vStartDate:vEndDate].signals,\
                                                validationDict[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int),\
                                                unfilteredData.Close, minFcst=PRT['horizon'] , DD95_limit =PRT['DD95_limit'] )
            model = [validationDict[validationPeriod].iloc[-1].system.split('_')[4],\
                            [m[1] for m in models if m[0] ==validationDict[validationPeriod].iloc[-1].system.split('_')[4]][0]]
            metaData['validationPeriod'] = validationPeriod
            #metaData['params'] =model[1]
            vCurve_metrics = update_report(vCurve_metrics, filterName,\
                                                validationDict[validationPeriod].ix[vStartDate:vEndDate].signals.values.astype(int), \
                                                dataSet.ix[validationDict[validationPeriod].ix[vStartDate:vEndDate].index].signal.values.astype(int),\
                                                unfilteredData.gainAhead,\
                                                validationDict[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int), model,\
                                                metaData,CAR25_oos)
    else:
        vCurve_metrics = init_report()
        for validationPeriod in validationDict:
            vCurve = ticker+' Validation Period '+str(validationPeriod)
            if showAllCharts:
                compareEquity(validationDict[validationPeriod].ix[vStartDate:vEndDate],vCurve)
            if scorePath is not None:
                validationDict[validationPeriod].to_csv(equityStatsSavePath+vCurve+'_'+\
                                                                str(validationDict[validationPeriod].index[0]).replace(':','')+'_to_'+\
                                                                str(validationDict[validationPeriod].index[-1]).replace(':','')+'.csv')
            CAR25_oos = CAR25_df_min(vCurve,validationDict[validationPeriod].ix[vStartDate:vEndDate].signals,\
                                                validationDict[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int),\
                                                unfilteredData.Close, minFcst=PRT['horizon'] , DD95_limit =PRT['DD95_limit'] )
            model = [validationDict[validationPeriod].iloc[-1].system.split('_')[4],\
                            [m[1] for m in models if m[0] ==validationDict[validationPeriod].iloc[-1].system.split('_')[4]][0]]
            #metaData['model'] = model[0]
            #metaData['params'] =model[1]
            metaData['validationPeriod'] = validationPeriod
            vCurve_metrics = update_report(vCurve_metrics, filterName,\
                                                validationDict[validationPeriod].ix[vStartDate:vEndDate].signals.values.astype(int), \
                                                dataSet.ix[validationDict[validationPeriod].ix[vStartDate:vEndDate].index].signal.values.astype(int),\
                                                unfilteredData.gainAhead,\
                                                validationDict[validationPeriod].ix[vStartDate:vEndDate].prior_index.values.astype(int), model,\
                                                metaData,CAR25_oos)
            
    #score models
    scored_models, bestModelParams = directional_scoring(vCurve_metrics,filterName)
    #bestModelParams['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    if scorePath is not None:
        scored_models.to_csv(scorePath+version+'_'+ticker+'.csv')
        
    print '\nBest Validation Period Found:', bestModelParams.validationPeriod
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    if showAllCharts:
        BestEquity = calcEquity_df(validationDict[bestModelParams.validationPeriod][['signals','gainAhead']],\
                                            bestModelParams.C25sig, leverage = validationDict[bestModelParams.validationPeriod].safef.values)
    timenow = dt.now(timezone('US/Eastern'))
    lastBartime = timezone('US/Eastern').localize(dataSet.index[-1].to_datetime())
    #adjust cycletime if weekend
    weekday = dt.now(timezone('US/Eastern')).weekday()
    if weekday == 5 or weekday ==6:
        cycleTime = round(((time.time() - start_time)/60),2)
    else:
        cycleTime = (timenow-lastBartime).total_seconds()/60
    bestModelParams = bestModelParams.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
    bestModelParams = bestModelParams.append(pd.Series(data=lastBartime.strftime("%Y%m%d %H:%M:%S %Z"), index=['lastBarTime']))
    bestModelParams = bestModelParams.append(pd.Series(data=cycleTime, index=['cycleTime']))
    
    print version_, 'Saving Params..'
    files = [ f for f in listdir(bestParamsPath) if isfile(join(bestParamsPath,f)) ]
    if version+'_'+ticker + '.csv' not in files:
        BMdf = pd.concat([bestModelParams,bestModelParams],axis=1).transpose()
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'.csv', index=False)
    else:
        BMdf = pd.read_csv(bestParamsPath+version+'_'+ticker+'.csv').append(bestModelParams, ignore_index=True)
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'.csv', index=False)
    
    print version_, 'Saving Model..'
    joblib.dump(BSMdict[bestModelParams.validationPeriod], modelSavePath+version_+'_'+ticker+'.joblib',compress=3)
    
    print version_, 'Saving Signals..'      
    #init file
    #BMdict[bestModelParams.validationPeriod].tail().to_csv(signalPath + version+'_'+ ticker + '.csv')
    files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
    
    #old version file
    if version+'_'+ ticker + '.csv' not in files:
        signalFile = BMdict[bestModelParams.validationPeriod].tail()
        signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker + '.csv', index_col=['dates'])
        addLine = BMdict[bestModelParams.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = BMdict[bestModelParams.validationPeriod].iloc[-1].name
        signalFile = signalFile.append(addLine)
        signalFile.to_csv(signalPath + version+'_'+ ticker + '.csv', index=True)
        
    #new version file
    if version_+'_'+ ticker + '.csv' not in files:
        #signalFile = BMdict[bestModelParams.validationPeriod].iloc[-2:]
        addLine = BMdict[bestModelParams.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = BMdict[bestModelParams.validationPeriod].iloc[-1].name
        signalFile = BMdict[bestModelParams.validationPeriod].iloc[-2:-1].append(addLine)
        signalFile.index.name = 'dates'
        signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker + '.csv', index_col=['dates'])
        addLine = BMdict[bestModelParams.validationPeriod].iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = BMdict[bestModelParams.validationPeriod].iloc[-1].name
        signalFile = signalFile.append(addLine)
        signalFile.to_csv(signalPath + version_+'_'+ ticker + '.csv', index=True)
    
    print '\n'+version, 'Next Signal:'
    print BMdict[bestModelParams.validationPeriod].iloc[-1].system
    print BMdict[bestModelParams.validationPeriod].drop(['system'],axis=1).iloc[-1]
    
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    print 'Finished Validation Period Parameter Search!'
    
    ##############################################################
