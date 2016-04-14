import numpy as np
import pandas as pd
import scipy as sp
import datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import cluster, covariance, manifold
from sklearn.qda import QDA
import talib as ta
import re
from dateutil.parser import parse
#import keras as k
#import tensorflow as tf

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    def addFeatures(dataframe, adjclose, returns, n):
        """
        operates on two columns of dataframe:
        - n >= 2
        - given Return_* computes the return of day i respect to day i-n. 
        - given AdjClose_* computes its moving average on n days
    
        """
        
        return_n = adjclose[9:] + "Time" + str(n)
        dataframe[return_n] = dataframe[adjclose].pct_change(n)
        
        roll_n = returns[7:] + "RolMean" + str(n)
        dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)
        
    def applyRollMeanDelayedReturns(datasets, delta):
        """
        applies rolling mean and delayed returns to each dataframe in the list
        """
        for dataset in datasets:
            columns = dataset.columns    
            adjclose = columns[-2]
            returns = columns[-1]
            for n in delta:
                addFeatures(dataset, adjclose, returns, n)
        
        return datasets    
    
    def mergeDataframes(datasets, index, cut):
        """
        merges datasets in the list 
        """
        #subset = []tion
        subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
        
        first = subset[0].join(subset[1:], how = 'outer')
        finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
        finance = finance[finance.index > cut]
        return finance
    
    def applyTimeLag(dataset, lags, delta):
        """
        apply time lag to return columns selected according  to delta.
        Days to lag are contained in the lads list passed as argument.
        Returns a NaN free dataset obtained cutting the lagged dataset
        at head and tail
        """
        
        dataset.Return_Out = dataset.Return_Out.shift(-1)
        maxLag = max(lags)
    
        columns = dataset.columns[::(2*max(delta)-1)]
        for column in columns:
            for lag in lags:
                newcolumn = column + str(lag)
                dataset[newcolumn] = dataset[column].shift(lag)
    
        return dataset.iloc[maxLag:-1,:]
    
    def prepareDataForClassification(dataset, start_test):
        """
        generates categorical output column, attach to dataframe 
        label the categories and split into train and test
        """
        le = preprocessing.LabelEncoder()
        
        dataset['UpDown'] = dataset['Return_Out']
        dataset.ix[dataset.UpDown >= 0,'UpDown'] = 'Up'
        dataset.ix[dataset.UpDown < 0,'UpDown'] = 'Down'
        dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
        
        print 'Dropping Data Missing: ', count_missing(dataset)
        dataset=dataset.replace([np.inf, -np.inf], np.nan)
        dataset=dataset.fillna(method='pad')
        dataset=dataset.fillna(method='bfill')
        #.dropna(subset=dataset.columns, how="all")
        print 'Data Missing: ', count_missing(dataset)
        features = dataset.columns[1:-1]
        X = dataset[features]    
        y = dataset.UpDown    
        
        X_train = X[X.index < start_test]
        y_train = y[y.index < start_test]              
        
        X_test = X[X.index >= start_test]    
        y_test = y[y.index >= start_test]
        
        return X_train, y_train, X_test, y_test   
        
    def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
        """
        performs classification on daily returns using several algorithms (method).
        method --> string algorithm
        parameters --> list of parameters passed to the classifier (if any)
        fout --> string with name of stock to be predicted
        savemodel --> boolean. If TRUE saves the model to pickle file
        """
       
        if method == 'RF':   
            return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
            
        elif method == 'KNN':
            return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
        elif method == 'SVM':   
            return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
        elif method == 'ADA':
            return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
        elif method == 'GTB': 
            return performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
        elif method == 'QDA': 
            return performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
            
    def performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
        global savePath
        print 'RFClass ' 
        
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        clf.fit(X_train, y_train)
           
        accuracy = clf.score(X_test, y_test)
        print 'Accuracy:',accuracy
        return clf
        
    def performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
        """
        KNN binary Classification
        """
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train) 
        
        accuracy = clf.score(X_test, y_test)
        
        print 'Accuracy:',accuracy
        return clf
        
    def performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
        """
        SVM binary Classification
        """
        c = parameters[0]
        g =  parameters[1]
        clf = SVC()
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_test, y_test)
        
        print 'Accuracy:',accuracy
        return clf
        
    def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
        """
        Ada Boosting binary Classification
        """
        n = parameters[0]
        l =  parameters[1]
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)  
        
        accuracy = clf.score(X_test, y_test)
        
        print 'Accuracy:',accuracy
        return clf
        
    def performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
        """
        Gradient Tree Boosting binary Classification
        """
        clf = GradientBoostingClassifier(n_estimators=100)
        clf.fit(X_train, y_train)  
        
        accuracy = clf.score(X_test, y_test)
        
        print 'Accuracy:',accuracy
        return clf
        
    def performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
        """
        Quadratic Discriminant Analysis binary Classification
        """
        def replaceTiny(x):
            if (abs(x) < 0.0001):
                x = 0.0001
        
        X_train = X_train.apply(replaceTiny)
        X_test = X_test.apply(replaceTiny)
        
        clf = QDA()
        clf.fit(X_train, y_train)
    
        accuracy = clf.score(X_test, y_test)
        
        print 'Accuracy:',accuracy
        return clf
    
    def getPredictionFromBestModel(bestdelta, bestlags, fout, cut, start_test, dataSets, parameters):
        """
        returns array of prediction and score from best model.
        """
        
        (X_train, y_train, X_test, y_test)=dataPrep(bestdelta, bestlags, fout, cut, start_test, dataSets, parameters)
        model = performClassification(X_train, y_train, X_test, y_test, 'RF', parameters, fout, False)
        
        #with open(parameters[0], 'rb') as fin:
        #    model = cPickle.load(fin)        
            
        return model.predict(X_test), model.score(X_test, y_test)
    
    def dataPrep(maxdelta, maxlag, fout, cut, start_test, dataSets, parameters):
        lags = range(2, maxlag) 
        
        delta = range(2, maxdelta) 
        print 'Delta days accounted: ', max(delta)
        datasets = applyRollMeanDelayedReturns(dataSets, delta)
        finance = mergeDataframes(datasets, 6, cut)
        print 'Size of data frame: ', finance.shape
        print 'Number of NaN after merging: ', count_missing(finance)
        finance = finance.interpolate(method='linear')
        print 'Number of NaN after time interpolation: ', count_missing(finance)
        finance = finance.fillna(finance.mean())
        print 'Number of NaN after mean interpolation: ', count_missing(finance)    
        finance = applyTimeLag(finance, lags, delta)
        print 'Number of NaN after temporal shifting: ', count_missing(finance)
        print 'Size of data frame after feature creation: ', finance.shape
        
        
        X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)
        return (X_train, y_train, X_test, y_test)
        
    def count_missing(df):
        res=len(df) - df.count()
        print 'Null Values:', df.isnull().sum().sum()
        #print 'Inf Values:', df.isinf().sum().sum()
        #for r in res:
        #    if r > 0:
        #        print r
        #return res
        return sum(np.array(res))
    
    def loadDatasets(symlist, fout, parameters):
        interval=parameters[1]
        symbol_dict=dict()
        for symbol in symlist:
            fsym=symbol
            if not re.search(fsym,fout):
                sym=symbol
                symbol_dict[sym]=sym
                print sym
        symbols, names = np.array(list(symbol_dict.items())).T
        
        out =  get_quote(symlist, fout, 'Out', True, parameters)
        data = list()
        for symbol in symbols:
            dataFrame=get_quote(symlist, symbol, symbol, True, parameters)
            if dataFrame.shape[0]>0:
                data.append(dataFrame)
            
        dataSet=list()
        dataSet.append(out)
        dataSet.extend(data)
        return dataSet
    
    def get_quote(symbols, sym, colname, addParam, parameters):
        idx=symbols.index(sym)
        dataSet=pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Vol','Oi','P','R','Rinfo'])
        dataSet['Date']=DATE
        dataSet['Date']=[parse(str(x)) for x in dataSet['Date']]
        dataSet['Open']=OPEN[:,idx]
        dataSet['High']=HIGH[:,idx]
        dataSet['Low']=LOW[:,idx]
        dataSet['Close']=CLOSE[:,idx]
        dataSet['Vol']=VOL[:,idx]
        dataSet['Oi']=OI[:,idx]
        dataSet['P']=P[:,idx]
        dataSet['R']=R[:,idx]
        dataSet['Rinfo']=RINFO[:,idx]
        dataSet=dataSet.set_index('Date')
            
        dataSet.index=pd.to_datetime(dataSet.index)
        dataSet=dataSet.sort_index()    
        #if parameters[2] > 0:
        #    dataSet=dataSet.iloc[:-parameters[2]].sort_index().copy();
            
        if addParam:
            
            ###### Future #######
            data=dataSet.iloc[-1].copy()
            data=dataSet.reset_index().iloc[-1].copy()
            #data['Open']=data['Close']
            data['Close']=data['Open']
            if parameters[1] == '30m_': 
                data['Date']=data['Date']+datetime.timedelta(minutes=30)
            elif parameters[1] == '1h_': 
                data['Date']=data['Date']+datetime.timedelta(hours=1)
            elif parameters[1] == '10m_': 
                data['Date']=data['Date']+datetime.timedelta(minutes=10)
            elif parameters[1] == '1 min_': 
                data['Date']=data['Date']+datetime.timedelta(minutes=1)
            else:
                data['Date']=data['Date']+datetime.timedelta(days=1)
            ###### DataSet ######
            dataSet=dataSet.reset_index().append(data).set_index('Date')
            #dataSet.columns.values[-1] = 'AdjClose'
            dataSet['AdjClose']=dataSet['Close']
            #dataSet['Open']=dataSet['Open']
            dataSet.columns = dataSet.columns + '_' + colname
            dataSet['Return_%s' %colname] = dataSet['AdjClose_%s' %colname].pct_change()
        #dataSet=dataSet.ix[dataSet.index[-1] - datetime.timedelta(days=10):]
        print 'Loaded '+ sym + ' ' + str(dataSet.shape[0]) + ' Rows'
        
        return dataSet
        
    symbol = 'ES'
    file='F_ES'
    bestModel=''
    lookback=1
    start_period = parse(str(DATE[0]))
    start_test = parse(str(DATE[-1])) - datetime.timedelta(days=lookback*2+14)  
    end_period = parse(str(DATE[-1]))
    interval='F_'
    parameters=list()
    parameters.append(bestModel)    
    parameters.append(interval)
    parameters.append(lookback)
    symbols=settings['markets']
    nMarkets=CLOSE.shape[1]
    pos=np.zeros((1,nMarkets))
    # skip until recent date
    #if not end_period > datetime.datetime.now() - datetime.timedelta(days=10):
    #    return pos, settings
    #print DATE, len(DATE), CLOSE[:,0],len(CLOSE[:,0])
    dataSets = loadDatasets(symbols, file, parameters)
    
    
    # skip if data missing
    for dataset in dataSets:
        #dataset=dataset.replace([np.inf, -np.inf], np.nan)
        #.dropna(subset=dataSet.columns, how="all")
        dataset=dataset.replace([np.inf, -np.inf], np.nan)
        dataset=dataset.fillna(method='pad')
        dataset=dataset.fillna(method='bfill')
        print 'Missing Data: ', count_missing(dataset)
        #if sum(np.array(count_missing(dataSet))) > 0:
        #    return pos, settings
    
    bData=dataSets
    
    # start prediction
    idx=symbols.index(file)
    print 'Training with sample up to: ' + str(start_test)
    print 'Creating Backtest Signal Up To: ' + str(bData[0].index[-2])
    print 'Last Date: ',bData[0].index[-2], ' Last Open: ',str(bData[0]['Open_Out'][-2]),  ' Last Close: ', bData[0]['Close_Out'][-2], ' Providing Look Future Data: Open ',bData[0]['Open_Out'][-1],' Close ',bData[0]['Close_Out'][-1]
    print 'Date:',DATE[-1],' Open: ', OPEN[-1][idx], ' Close: ', CLOSE[-1][idx], ' OI:', OI[-1][idx], ' Size:',len(DATE)
    prediction = getPredictionFromBestModel(9, 9, file, start_period, start_test, bData, parameters)
    nextSignal=prediction[0][-1]
    if nextSignal == 0:
        nextSignal=-1
    print 'Next Signal: ', nextSignal
    pos[0][idx]=nextSignal
    
    print symbols
    print pos
    
    return pos, settings



##### Do not change this function definition #####
def mySettings():
    '''Define your market list and other settings here.

    The function name "mySettings" should not be changed.

    Default settings are shown below.'''

    # Default competition and evaluation mySettings
    settings= {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL', \
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C', \
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',\
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE', \
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM', \
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON', \
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM', \
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP', \
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets']  = ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CD',  \
    'F_CL', 'F_DJ', 'F_EC', 'F_ES', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_LC', \
    'F_LN', 'F_NG', 'F_NQ', 'F_RB', 'F_S', 'F_SF', 'F_SI', 'F_SM', 'F_SP', \
    'F_TY', 'F_US', 'F_W', 'F_YM']

    settings['lookback']= 6000
    settings['budget']= 10**6
    settings['slippage']= 0.05
    settings['participation']= 0.1

    return settings
