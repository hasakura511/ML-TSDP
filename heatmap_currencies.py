import sys
import copy
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns

if len(sys.argv)==1:
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    savePath =  'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/currencies_heatmap' 
    pairPath='D:/ML-TSDP/data/'
else:
    savePath = '.data/results/currencies_heatmap'
    dataPath = './data/from_IB/'
    pairPath='./data/'

with open(pairPath+'currencies.txt') as f:
    currencyPairs = f.read().splitlines()
    
lookback=130
currencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
barSizeSetting='30m'

cDict=pd.DataFrame()
for currency in currencies:
    #cDict[currency]=pd.DataFrame()
    pairList=[pair for pair in currencyPairs if currency in pair[0:3] or currency in pair[3:6]]
    #files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    for pair in pairList:    
        data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)[-(lookback+1):]
        if currency in pair[0:3]:
            #print pair[3:6],currency,data.Close.pct_change(periods=lookback)[-1]*100
            cDict.set_value(pair[3:6],currency,-data.Close.pct_change(periods=lookback)[-1]*100)
        else:
            #print pair[0:3],currency,-data.Close.pct_change(periods=lookback)[-1]*100
            #pair2=pair[3:6]+pair[0:3]
            cDict.set_value(pair[0:3], currency,data.Close.pct_change(periods=lookback)[-1]*100)
            
for currency in currencies:
    cDict.set_value(currency,'Avg',cDict.ix[currency].dropna().mean())
rankByMean=cDict['Avg'].sort_values(ascending=False)
print data.index[0],'to',data.index[-1]
print cDict
fig,ax = plt.subplots(figsize=(8,8))
sns.heatmap(ax=ax,data=cDict)
#plt.pcolor(cDict)
#plt.yticks(np.arange(0.5, len(cDict.index), 1), cDict.index)
#plt.xticks(np.arange(0.5, len(cDict.columns), 1), cDict.columns)
if savePath != None:
    print 'Saving '+savePath+'.png'
    fig.savefig(savePath+'.png', bbox_inches='tight')
    
plt.show()
print 'Overall Rank'
print rankByMean

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
print 'buyHold',len(buyHold),buyHold
print 'sellHold',len(sellHold),sellHold

with open(pairPath+'buyHold_currencies.txt','w') as f:
    for pair in buyHold:
      f.write("%s\n" % pair)
      
with open(pairPath+'sellHold_currencies.txt','w') as f:
    for pair in sellHold:
      f.write("%s\n" % pair)