import socket   
import select
import sys
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import subprocess
import json
import time
from pandas.io.json import json_normalize
import pandas as pd
import threading
#from btapi.get_signal import get_v4signal
#from btapi.get_hist_btcharts import get_bthist
#from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv
#from seitoolz.paper import adj_size
from suztoolz.debug_system_v43C_30min_func import runv4
import sys
#import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
#import websocket
from suztoolz.display import offlineMode
import seitoolz.bars as bars
from multiprocessing import Process, Queue
import os
from pytz import timezone
from dateutil.parser import parse

from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
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
    


def get_last_bars_debug(currencyPairs, ylabel, callback, **kwargs):
    global tickerId
    global lastDate
    minPath = kwargs.get('minPath','./data/bars/')
    #while 1:
    try:
        SST=pd.DataFrame()
        symbols=list()
        returnData=False
        for i,ticker in enumerate(currencyPairs):
            pair=ticker
            minFile=minPath+pair+'.csv'
            symbol = pair
            #print minFile
            if os.path.isfile(minFile):
                logging.info(str(i)+' loading '+minFile)
                dta=pd.read_csv(minFile).iloc[-1]
                date=dta['Date']
                
                eastern=timezone('US/Eastern')
                date=parse(date).replace(tzinfo=eastern)
                timestamp = time.mktime(date.timetuple())
                dta.Date = date
                #data=pd.DataFrame()
                #data['Date']=date
                #data[symbol]=dta[ylabel]
                #print dta[ylabel]
                #print data, date
                #data=data.set_index('Date') 
                dta.name = dta.Date
                
                #if len(SST.index.values) < 1:
                #    SST=dta
                #else:
                 #   SST=SST.join(dta)
                    
                if not lastDate.has_key(symbol):
                    lastDate[symbol]=timestamp
                                           
                if lastDate[symbol] < timestamp:
                    returnData=True
                symbols.append(symbol)
                        
            #if returnData:
            #data=SST
            #data=data.set_index('Date')
            #data=data.fillna(method='pad')
        callback(dta, symbols)
            #time.sleep(20)
    except Exception as e:
        logging.error("get_last_bar", exc_info=True)
        
def get_bars(pairs, interval):
    #global SST
    #global start_time
    mypairs=list()
    for pair in pairs:
        mypairs.append(interval + pair)
        
    if debug:
        get_last_bars_debug(mypairs, 'Close', onBar,minPath=minPath)
    else:
        bars.get_last_bars(mypairs, 'Close', onBar)
        #get_last_bars_debug(mypairs, 'Close', onBar)

def onBar(bar, symbols):
    global start_time
    global gotbar
    global pairs
    bar = bar.iloc[-1]
    logging.info('received '+str(symbols)+str(bar))
    if not gotbar.has_key(bar['Date']):
        gotbar[bar['Date']]=list()
    #print bar['Date'], gotbar[bar['Date']]
    for symbol in symbols:
        if symbol not in gotbar[bar['Date']]:
            gotbar[bar['Date']].append(symbol)
    #gotbar[bar['Date']]=[i for sublist in gotbar[bar['Date']] for i in sublist]
    logging.info(str(bar['Date'])+ str(gotbar[bar['Date']])+ str(len(gotbar[bar['Date']]))+'bars '+ str(len(pairs))+'pairs')
    #global SST
    #SST = SST.combine_first(bar).sort_index()
    #if debug:
    
    if len(gotbar[bar['Date']])==len(livePairs):
    #if len([p for p in gotbar[bar['Date']] if p in livePairs]) == len(livePairs):
        #print gotbar[bar['Date']]
        start_time2 = time.time()
        for sym in gotbar[bar['Date']]:
            logging.info('')
            logging.info(sym+' timenow: '+dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z"))
            runPair_v4([sym])
        logging.info( 'All signals created for bar '+str(bar['Date']))
        logging.info('Runtime: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes' ) 
        logging.info('Last bar time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes\n' ) 
        start_time = time.time()
    #else:   
    #    if len(gotbar[bar['Date']])==len(pairs):
    #        print  len(gotbar[bar['Date']]), 'bars collected for', bar['Date'],'running systems..'
     #       for sym in gotbar[bar['Date']]:
                #print sym
     #           runPair_v4(sym)
                
    
        
def runPair_v4(pair):
    ticker = pair[0].split('_')[1]
    #version = 'v4'
    #version_ = 'v4.3C'
    runDPS = False
    runData = {'version':version, 'version_':version_,'asset':asset,'asset':asset,\
                    'barSizeSetting':barSizeSetting,'currencyPairs':currencyPairs,'debug':debug,'bias':bias,\
                    'volatilityThreshold':volatilityThreshold, 'validationSetLength':validationSetLength,'livePairs':livePairs,\
                    'ticker':ticker, 'dataPath':dataPath,'signalPath':signalPath,'chartSavePath':chartSavePath,\
                    'showCharts':showCharts, 'showFinalChartOnly':showFinalChartOnly,'showIndicators':showIndicators,\
                    'verbose':verbose, 'supportResistanceLB':supportResistanceLB,'nfeatures':nfeatures,\
                    'minDatapoints' :minDatapoints, 'metric':metric, 'metric2' :metric2,\
                    'addAuxPairs' :addAuxPairs, 'perturbData' :  perturbData, 'perturbDataPct' :perturbDataPct,\
                    'useDPSsafef' :useDPSsafef, 'PRT' :  PRT, 'maxlb' :maxlb,'maxReadLines':maxReadLines,\
                    'initialEquity':initialEquity,'nodpsSafef':nodpsSafef, 'fs_models':fs_models,\
                    'short_models':short_models,'noAuxPairs_models':noAuxPairs_models,'auxPairs_models':auxPairs_models,\
                    'outer_zz_std':outer_zz_std,'inner_zz_std':inner_zz_std}
                        
                
        
    try:
        with open ('/logs/'+ticker +version+ '.log','a') as f:
            orig_stdout = sys.stdout
            sys.stdout = f
            print 'Starting '+version+': ' + ticker
            if ticker not in livePairs:
                offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
            #f.write('Starting '+version+': ' + ticker)         
            #ferr.write('Starting '+version+': ' + ticker)
            signal=runv4(runData)
            print signal
            logging.info(version+' '+' signal '+str(signal.signals)+ ' safef '+str(signal.safef)+' CAR25 '+str(signal.CAR25))
            #logging.info(signal.system)
            #subprocess.call(['python','debug_system_v4.3C_30min.py',ticker,'1'], stdout=f, stderr=ferr)
            #f.close()
            #ferr.close()

        sys.stdout = orig_stdout
        #runPair_v2(pair, dataSet)
    except Exception as e:
        	 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
        	 #ferr.write(e)
        	 #ferr.close()
        	 logging.error("something bad happened", exc_info=True)

             
def runThreads():
    global start_time
    start_time = time.time()
    threads = []
    for pair in pairs:
        sig_thread = threading.Thread(target=get_bars, args=[[pair], barSizeSetting+'_'])
        sig_thread.daemon=True
        threads.append(sig_thread)
        sig_thread.start()
        
    if debug==False:
        while 1:
            time.sleep(100)




start_time = time.time()
lastDate={}
tickerId=1
gotbar=dict()
if __name__ == "__main__":
    debug=False
    #system parameters
    version = 'v4'
    version_ = 'v4.3'
    asset = 'FX'
    #filterName = 'DF1'
    #data_type = 'ALL'
    pairs =currencyPairs=livePairs=[sys.argv[1]]
    barSizeSetting=sys.argv[2]
    bias = [sys.argv[3]]
    volatilityThreshold=float(sys.argv[4])
    supportResistanceLB = int(sys.argv[5])
    validationSetLength =int(sys.argv[6])
    
    ticker =livePairs[0]
    logging.basicConfig(filename='/logs/runsystem_'+ticker+'v4onBar.log',level=logging.DEBUG)
    
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    #this gets overwritten
    chartSavePath = './data/results/'+version+'_'+ticker
    
    #display params
    showCharts=False
    showFinalChartOnly=True
    showIndicators = False
    verbose=False

    #Model Parameters



    #for PCA/KBest
    nfeatures = 10
    #if major low/high most recent index. minDatapoints sets the minimum is period.
    minDatapoints = 2
    #set to 1 for live
    #system selection metric
    #metric = 'CAR25'
    metric = 'netEquity'
    #metric for signal
    metric2='netEquity'
    #adds auxilary pair features
    addAuxPairs = False

    #robustness
    perturbData = False
    perturbDataPct = 0.0002


    #dps
    #save to signal file
    useDPSsafef=False
    #personal risk tolerance parameters
    PRT={}
    #ie goes to charts, car25 scoring (affected by commissions)
    PRT['initial_equity'] = 100000
    #fcst horizon(bars) for dps.  for training,  horizon is set to nrows.  for validation scoring nrows. 
    PRT['horizon'] = 50
    #safef set to dd95 where limit is met. e.g. for 50 bars, set saef to where 95% of the mc eq curves' maxDD <=0.01
    PRT['DD95_limit'] = 0.01
    PRT['tailRiskPct'] = 95
    #rounds safef and safef cannot go below this number. if set to None, no rounding
    PRT['minSafef'] =1
    #no dps safef
    PRT['nodpsSafef'] =1
    #dps max limit
    PRT['maxSafef'] = 2
    #safef=minSafef if CAR25 < threshold
    PRT['CAR25_threshold'] = 0
    #PRT['CAR25_threshold'] = -np.inf


    #needs 2x srlookback to prime indicators. 
    maxlb = supportResistanceLB*2
    maxReadLines = validationSetLength+maxlb
    initialEquity=PRT['initial_equity']
    nodpsSafef=PRT['nodpsSafef']
    #model selection
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    fs_models = [
             ('PCA'+str(nfeatures),PCA(n_components=nfeatures)),\
             ('SelectKBest'+str(nfeatures),SelectKBest(f_classif, k=nfeatures))\
             ]
    short_models = [
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             ("GNBayes",GaussianNB()),\
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
             #("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=2, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                 #], voting='hard', weights=None)),
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
    noAuxPairs_models = [
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             #("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
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
                 ("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=minDatapoints, weights='uniform')),\
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
                    #], voting='soft', weights=None)),
             ]
             
    auxPairs_models = [
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
             #("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=5, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                    #], voting='hard', weights=None)),
             ("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 ("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 ("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                     ], voting='soft', weights=None)),
             ]

    inner_zz_std =2
    outer_zz_std=4
    
    if len(sys.argv) <7:
        print 'syntax is python Q_Q.py pair barsize bias volatility lookback vlength  '
    else:            
        #if sys.argv[1] == 'single':  
        while 1:
            start_time = time.time()
            logging.info( 'starting single thread mode for '+str(barSizeSetting)+' bars '+str(len(livePairs))+' pairs.')
            logging.info(str(livePairs) )
            get_bars(livePairs,barSizeSetting+'_')
            #time.sleep(100)

    

