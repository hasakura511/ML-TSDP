import sys
import copy
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt

if len(sys.argv)==1:
    dataPath = 'D:/ML-TSDP/data/from_MT4/'
    savePath =  'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    pairPath='D:/ML-TSDP/data/'
else:
    savePath = './data/results/'
    dataPath = './data/from_MT4/'
    pairPath='./data/'

with open(pairPath+'currencies.txt') as f:
    currencyPairs = f.read().splitlines()
verbose=False
lookback=1
currencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
#currencies = ['EUR', 'GBP', 'JPY', 'USD']
barSizeSetting='1d'
for i in range(2,-1,-lookback):
    #startDate=dt(2016, 5, i,0,00)
    cMatrix=pd.DataFrame()
    #cMatrix2=pd.DataFrame()
    for currency in currencies:
        #lookback=i
        #cMatrix[currency]=pd.DataFrame()
        pairList=[pair for pair in currencyPairs if currency in pair[0:3] or currency in pair[3:6]]
        pairList =[pair for pair in pairList if pair[0:3] in currencies and pair[3:6] in currencies]
        #files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
        for pair in pairList:    
            #print -(i+1), -(1-lookback+1)
			if i ==0:
				data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)[-(i+2):]
			else:
				data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)[-(i+2):-(i-lookback+1)]
			data.index = data.index.to_datetime()
            #lookback=data.ix[startDate:].shape[0]
			if currency in pair[0:3]:
				#print pair[3:6],currency,data.Close.pct_change(periods=lookback)[-1]*100
				#cMatrix.set_value(pair[3:6],currency,-data.Close.pct_change(periods=lookback)[-1]*100)
				cMatrix.set_value(pair[3:6],currency,-data.Close.pct_change(periods=lookback)[-1]*100)
			else:
				#print pair[0:3],currency,-data.Close.pct_change(periods=lookback)[-1]*100
				#pair2=pair[3:6]+pair[0:3]
				#cMatrix.set_value(pair[0:3], currency,data.Close.pct_change(periods=lookback)[-1]*100)
				cMatrix.set_value(pair[0:3], currency,data.Close.pct_change(periods=lookback)[-1]*100)
                
    for currency in currencies:
        cMatrix.set_value(currency,'Avg',cMatrix.ix[currency].dropna().mean())
    #cMatrix=cMatrix.sort_values(by='Avg', ascending=False).fillna(0)
    cMatrix=cMatrix.fillna(0)
    rankByMean=cMatrix['Avg']
    '''
    with open(savePath+'currencies_1.html','w') as f:
        f.write(str(startDate)+' to '+str(data.index[-1]))
        
    cMatrix.to_html(savePath+'currencies_4.html')
    '''
    #print data.index[0],'to',data.index[-1]
    #print cMatrix
    fig,ax = plt.subplots(figsize=(8,8))
    sns.heatmap(ax=ax,data=cMatrix)
    #ax.set_title(str(data.ix[startDate].name)+' to '+str(data.index[-1]))
    startDate=data.index[0]
    ax.set_title(str(data.ix[startDate].name)+' to '+str(data.index[-1]))
    #plt.pcolor(cMatrix)
    #plt.yticks(np.arange(0.5, len(cMatrix.index), 1), cMatrix.index)
    #plt.xticks(np.arange(0.5, len(cMatrix.columns), 1), cMatrix.columns)
    if savePath != None:
        print 'Saving '+savePath+'currencies_'+str(i+2)+'.png'
        fig.savefig(savePath+'currencies_'+str(i+2)+'.png', bbox_inches='tight')
        
    if len(sys.argv)==1:
        #print startDate,'to',data.index[-1]
        plt.show()


    ranking = rankByMean.index
    buyHold=[]
    sellHold=[]

    cplist = copy.deepcopy(currencyPairs)
    for currency in ranking:
        for i,pair in enumerate(cplist):
            #print pair
            if pair not in buyHold and pair not in sellHold:
                if currency in pair[0:3]:
                    #print i,'bh',pair
                    buyHold.append(pair)
                    #cplist.remove(pair)
                elif currency in pair[3:6]:
                    #print i,'sh',pair
                    sellHold.append(pair)
                    #cplist.remove(pair)
                #else:
                    #print i,currency,pair
    offline=[pair for pair in currencyPairs if pair not in buyHold+sellHold]
    if verbose:
        print startDate,'to',data.index[-1]
        print 'Overall Rank\n',rankByMean
        print 'buyHold',len(buyHold),buyHold
        print 'sellHold',len(sellHold),sellHold
        print 'offline',len(offline),offline
        
lastPrices=pd.DataFrame()
for pair in sorted(currencyPairs):
    data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0).iloc[-1]
    lastPrices.set_value(pair,data.name,data.Close)
lastPrices.to_csv(pairPath+'lastCurrencyPrices.csv')
'''
with open(pairPath+'buyHold_currencies.txt','w') as f:
    for pair in buyHold:
      f.write("%s\n" % pair)
      
with open(pairPath+'sellHold_currencies.txt','w') as f:
    for pair in sellHold:
      f.write("%s\n" % pair)

with open(pairPath+'offline_currencies.txt','w') as f:
    for pair in offline:
      f.write("%s\n" % pair)
'''