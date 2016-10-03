import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import time
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

start_time = time.time()

if len(sys.argv)==1:
    debug=True
    showPlots=True
    dataPath='D:/ML-TSDP/data/'
    portfolioPath = 'D:/ML-TSDP/data/portfolio/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
else:
    debug=False
    showPlots=False
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    savePath='./data/'
    savePath2 = './data/results/'

portfolioFilename = 'c2_v4futures_portfolio.csv'
tradeFilename='v4futures_c2trades.csv'
csidataFilename = 'futuresATR.csv'
#adjustments
adjDict={
            '@CT':100,
            '@JY':0.01,
            'QSI':0.01
            }

#entry trades
portfolioDF = pd.read_csv(portfolioPath+portfolioFilename)
portfolioDF['openedWhen'] = pd.to_datetime(portfolioDF['openedWhen'])
portfolioDF = portfolioDF.sort_values(by='openedWhen', ascending=False)
#exit trades
tradesDF = pd.read_csv(portfolioPath+tradeFilename)
tradesDF=tradesDF.drop(['expir','putcall','strike','symbol_description','underlying','markToMarket_time'], axis=1).dropna()
portfolioDF['closedWhen'] = pd.to_datetime(portfolioDF['closedWhen'])
tradesDF = tradesDF.sort_values(by='closedWhen', ascending=False)
#csi close data
futuresDF = pd.read_csv(dataPath+csidataFilename, index_col=0)
#csidata download at 8pm est
futuresDate = dt.strptime(futuresDF.index.name, '%Y-%m-%d %H:%M:%S').replace(hour=20)
#portfolioDF[portfolioDF['openedWhen']>=futuresDate]
slipDF = pd.DataFrame()
print 'sym', 'c2price', 'csiPrice', 'slippage'
for contract in portfolioDF.symbol.values:
    c2price=portfolioDF[portfolioDF.symbol ==contract].closing_price_VWAP.values[0]
    if contract in futuresDF.Contract.values:
        if contract[:-2] in adjDict.keys():
            csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
        else:
            csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
        slippage=(c2price-csiPrice)/csiPrice
        #print contract, c2price, csiPrice,slippage
        rowName = 'ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
        slipDF.set_value(rowName, 'c2price', c2price)
        slipDF.set_value(rowName, 'csiPrice', csiPrice)
        slipDF.set_value(rowName, 'slippage', slippage)
        slipDF.set_value(rowName, 'abs_slippage', abs(slippage))

if slipDF.shape[0] != portfolioDF.shape[0]:
    print 'Warning! Some values are mising'

filename='slippage_report_'+futuresDF.index.name.split()[0].replace('-','')+'.csv'
slipDF = slipDF.sort_values(by='abs_slippage', ascending=True)
slipDF.to_csv(savePath+'slippage_report.csv')
print 'Saved '+savePath+'slippage_report.csv'
slipDF.to_csv(savePath2+filename)
print 'Saved '+savePath2+filename

plt.figure(figsize=(8,13))
ax = slipDF.slippage.plot.barh(color='r', width=0.5)

#plt.axvline(0, color='k')
plt.title('Slippage '+futuresDF.index.name)
#plt.ylim(0,80)
#plt.xticks(np.arange(-1,1.25,.25))
plt.grid(True)
filename='futures_4'+'.png'
if savePath2 != None:
    plt.savefig(savePath2+filename, bbox_inches='tight')
    print 'Saved '+savePath2+filename
if len(sys.argv)==1 and showPlots:
    #print data.index[0],'to',data.index[-1]
    plt.show()
plt.close()
    
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()