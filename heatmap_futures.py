import copy
import pandas as pd
from os import listdir
from os.path import isfile, join
import sys
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

with open('./data/futures.txt') as f:
    futures = f.read().splitlines()
    
if len(sys.argv)==1:
    dataPath = 'D:/data/tickerData/'
    savePath =  'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    pairPath='D:/ML-TSDP/data/'
else:
    savePath = './data/results/'
    dataPath = './data/tickerData/'
    pairPath='./data/'
lookback=1
#currencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
#barSizeSetting='30m'
#dataPath='D:/data/tickerData/'
futuresMatrix=pd.DataFrame()
for contract in futures:
    data = pd.read_csv(dataPath+'F_'+contract+'.txt', index_col=0)[-(lookback+1):]
    #data = data.drop([' P',' R', ' RINFO'],axis=1)
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    #data.columns = ['Open','High','Low','Close','Volume','OI']
    #futuresMatrix[contract]=pd.DataFrame()
    #pairList=[pair for pair in futures if contract in pair[0:3] or contract in pair[3:6]]
    #files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    for contract2 in futures:    
        data2 = pd.read_csv(dataPath+'F_'+contract2+'.txt', index_col=0)[-(lookback+1):]
        #data2 = data2.drop([' P',' R', ' RINFO'],axis=1)
        data2.index = pd.to_datetime(data2.index,format='%Y%m%d')
        #data2.columns = ['Open','High','Low','Close','Volume','OI']
        #if contract in pair[0:3]:
            #print pair[3:6],contract,data.Close.pct_change(periods=lookback)[-1]*100
        futuresMatrix.set_value(contract,contract2,(data.Close/data2.Close).pct_change(periods=lookback)[-1]*100)
        #print futuresMatrix
        #else:
            #print pair[0:3],contract,-data.Close.pct_change(periods=lookback)[-1]*100
            #pair2=pair[3:6]+pair[0:3]
         #   futuresMatrix.set_value(pair[0:3], contract,-data.Close.pct_change(periods=lookback)[-1]*100)
for contract in futures:
    futuresMatrix.set_value(contract,'Avg',futuresMatrix.ix[contract].dropna().mean())
futuresMatrix=futuresMatrix.sort_values(by='Avg', ascending=False)
#rankByMean=futuresMatrix['Avg'].sort_values(ascending=False)

#with open(savePath+'futures_1.html','w') as f:
#    f.write(str(data.index[0])+' to '+str(data.index[-1]))
    
futuresMatrix.to_html(savePath+'futures_3.html')

#print futuresMatrix
fig,ax = plt.subplots(figsize=(13,13))
ax.set_title(str(data.index[0])+' to '+str(data.index[-1]))
sns.heatmap(ax=ax,data=futuresMatrix)
#plt.pcolor(futuresMatrix)
#plt.yticks(np.arange(0.5, len(futuresMatrix.index), 1), futuresMatrix.index)
#plt.xticks(np.arange(0.5, len(futuresMatrix.columns), 1), futuresMatrix.columns)
if savePath != None:
    print 'Saving '+savePath+'futures_2.png'
    fig.savefig(savePath+'futures_2.png', bbox_inches='tight')
    
if len(sys.argv)==1:
    print data.index[0],'to',data.index[-1]
    plt.show()
#print rankByMean

'''
ranking = rankByMean.index
buyHold=[]
sellHold=[]
cplist = copy.deepcopy(futures)
for contract in ranking:
    for i,pair in enumerate(cplist):
        #print pair
        if pair not in buyHold and pair not in sellHold:
            if contract in pair[0:3]:
                #print i,'bh',pair
                buyHold.append(pair)
                #cplist.remove(pair)
            elif contract in pair[3:6]:
                #print i,'sh',pair
                sellHold.append(pair)
                #cplist.remove(pair)
            #else:
                #print i,contract,pair
print 'buyHold',len(buyHold),buyHold
print 'sellHold',len(sellHold),sellHold
'''
