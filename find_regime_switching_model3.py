# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Wed Dec  2 06:10:48 2015

@author: hidemi


find the switching model
"""
import string
from os import listdir
from os.path import isfile, join
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import datetime
from pandas.core import datetools
import time
from suztoolz.transform import ratio
from suztoolz.loops import calcDPS2, calcEquity2, createBenchmark, createYearlyStats, CAR25_df
from suztoolz.display import displayRankedCharts
from sklearn.preprocessing import scale, robust_scale, minmax_scale
import warnings
warnings.filterwarnings('error')   
import talib as ta

start_time = time.time()
systemName = 'DoubleDF_ES'
ticker = 'F_ES'
nTopSystems = 1 #n top systems to show
numRSCharts = 3
emaLengths = [30]
#could be better to set to 0 when WindowLength is low.
rs_threshold = 0.0


#  Set the path for the csv file
#mypath = '/media/sf_Python/data/to_RS/'
dpsSavePath = '/media/sf_Python/data/from_DPS_to_RS/'
rsSavePath = '/media/sf_Python/data/from_RS/'
#filename = 'SST_F_ES_vf_VotingHard df_KNeighborsRegressor-distance_is250oos1_2014-01-02to2015-12-31_20160113074238.csv'
signal_type = 'ALL'
#metric to use for regime switching


PRT={}
PRT['DD95_limit'] = 0.20
PRT['tailRiskPct'] = 95
PRT['initial_equity'] = 1.0
PRT['horizon'] = 250
PRT['maxLeverage'] = 2
#PRT['CAR25_threshold'] = -20 #set safef to 0 if < car threshold

#for CAR25 calc
unfilteredDataFile = 'OHLCV_F_ES_TOX25_VotingHard_v2015-01-01to2017-01-01_20160226163023.csv'
file_path = '/media/sf_Python/data/from_vf_to_df/'
qt = pd.read_csv(file_path+unfilteredDataFile, index_col='dates')
#ALIGNMENT CHECK
#close = qt[[' CLOSE','gainAhead']]
close = qt[' CLOSE']
close.index = close.index.to_datetime()
close.name = 'Close'

#the files in the folder all need to be from the same time period
print 'Using',dpsSavePath,'to load files..'
aux_files = [ f for f in listdir(dpsSavePath) if isfile(join(dpsSavePath,f)) ]
print 'loading', len(aux_files), 'files..'
signals = pd.DataFrame()
DPS = {}
for i,f in enumerate(aux_files):
    print 'loading..', f[:-4]
    #new code uses dates
    if 'dates' in pd.read_csv(dpsSavePath+f).columns:
        DPS[f[:-4]] = pd.read_csv(dpsSavePath+f, index_col=['dates'])
    else:
        DPS[f[:-4]] = pd.read_csv(dpsSavePath+f, index_col=['Unnamed: 0'])

#change back zeros to 1's and -1's
for dps in DPS:
    #print dps
    if string.split(dps,'__')[1][0] == 's':
        DPS[dps].signals[DPS[dps].signals == 0] = 1
    elif string.split(dps,'__')[1][0] == 'l':
        DPS[dps].signals[DPS[dps].signals == 0] = -1
    else:
        pass
        
start = DPS[dps].index[0]
end = DPS[dps].index[-1]
startDate = DPS[dps].set_index(pd.DatetimeIndex(DPS[dps].index)).index[0]
endDate = DPS[dps].set_index(pd.DatetimeIndex(DPS[dps].index)).index[-1]

metrics = ['CAR25']
#simple ema switching model
for el in emaLengths:
    metrics.append('EMA'+str(el))
    
for el in emaLengths:
    print 'adding', 'CAR25 EMA'+str(el), 'to DPS' 
    for dps in DPS:
        c25ema = ta.EMA(DPS[dps].CAR25.values,el)
        #set nan values to CAR25
        c25ema[np.isnan(c25ema)] = DPS[dps].CAR25[:len(c25ema[np.isnan(c25ema)])].values
        DPS[dps]['EMA'+str(el)] = c25ema

dpsRunsByWL = {}
for dps in DPS:
    #find wl in string, split into words, get the first word, get the number after the 'wl'
    wl = float(string.split(dps[string.find(dps,'wl'):])[0][2:])
    if wl in dpsRunsByWL:
        dpsRunsByWL[wl].append(dps)
    else:
        #make a list
        dpsRunsByWL[wl] = [dps]


#find the system with the Highest CAR25 for that date
#sMaxCAR = pd.Series(index =sst_save.index)
RS= {}
if len(dpsRunsByWL) > 1:
    #create Regime switching for each windowlength using simple logic
    for metric in metrics:
        for wl in dpsRunsByWL:
            print 'Creating Regime Switching with windowlength',wl,'using',metric
            if metric != 'CAR25':
                systemWithMaxCAR = pd.DataFrame(columns=['system','signals','gainAhead','safef','CAR25',\
                           metric,'dd95','ddTol'], index =DPS[dps].index)
            else:
                systemWithMaxCAR = pd.DataFrame(columns=['system','signals','gainAhead','safef','CAR25',\
                           'dd95','ddTol'], index =DPS[dps].index)
            for i in range(0,DPS[dps].shape[0]):
                #init
                maxM = rs_threshold
                systemWithMaxCAR['system'].iloc[i] = metric,'less than threshold of',rs_threshold
                systemWithMaxCAR['signals'].iloc[i] = 0
                systemWithMaxCAR['gainAhead'].iloc[i] = DPS[dps]['gainAhead'].iloc[i]
                systemWithMaxCAR['safef'].iloc[i] = 0
                systemWithMaxCAR['CAR25'].iloc[i] = 0
                if metric != 'CAR25':
                    systemWithMaxCAR[metric].iloc[i] = 0
                systemWithMaxCAR['dd95'].iloc[i] = 0
                systemWithMaxCAR['ddTol'].iloc[i] = 0
                for dps in dpsRunsByWL[wl]:
                    if DPS[dps][metric].iloc[i] > maxM:
                        maxM = DPS[dps][metric].iloc[i]
                        systemWithMaxCAR['system'].iloc[i] = dps
                        systemWithMaxCAR['signals'].iloc[i] = DPS[dps]['signals'].iloc[i]
                        systemWithMaxCAR['gainAhead'].iloc[i] = DPS[dps]['gainAhead'].iloc[i]
                        systemWithMaxCAR['safef'].iloc[i] = DPS[dps]['safef'].iloc[i]
                        systemWithMaxCAR['CAR25'].iloc[i] = DPS[dps]['CAR25'].iloc[i]
                        if metric != 'CAR25':
                            systemWithMaxCAR[metric].iloc[i] = DPS[dps][metric].iloc[i]
                        systemWithMaxCAR['dd95'].iloc[i] = DPS[dps]['dd95'].iloc[i]
                        systemWithMaxCAR['ddTol'].iloc[i] = DPS[dps]['ddTol'].iloc[i]

                    #sMaxCAR.iloc[i] = maxM
            RS['RegimeSwitching_wl'+str(wl)+'_'+metric] = systemWithMaxCAR

    #create Regime switching for all windowlengths using simple logic
    for metric in metrics:
        print 'Creating Regime Switching for all windows using',metric
        if metric != 'CAR25':
            systemWithMaxCAR = pd.DataFrame(columns=['system','signals','gainAhead','safef','CAR25',\
                       metric,'dd95','ddTol'], index =DPS[dps].index)
        else:
            systemWithMaxCAR = pd.DataFrame(columns=['system','signals','gainAhead','safef','CAR25',\
                       'dd95','ddTol'], index =DPS[dps].index)
        for i in range(0,DPS[dps].shape[0]):
            #init
            maxM = rs_threshold
            systemWithMaxCAR['system'].iloc[i] = metric,'less than threshold of',rs_threshold
            systemWithMaxCAR['signals'].iloc[i] = 0
            systemWithMaxCAR['gainAhead'].iloc[i] = DPS[dps]['gainAhead'].iloc[i]
            systemWithMaxCAR['safef'].iloc[i] = 0
            systemWithMaxCAR['CAR25'].iloc[i] = 0
            if metric != 'CAR25':
                systemWithMaxCAR[metric].iloc[i] = 0
            systemWithMaxCAR['dd95'].iloc[i] = 0
            systemWithMaxCAR['ddTol'].iloc[i] = 0
            for dps in DPS:
                if DPS[dps][metric].iloc[i] > maxM:
                    maxM = DPS[dps][metric].iloc[i]
                    systemWithMaxCAR['system'].iloc[i] = dps
                    systemWithMaxCAR['signals'].iloc[i] = DPS[dps]['signals'].iloc[i]
                    systemWithMaxCAR['gainAhead'].iloc[i] = DPS[dps]['gainAhead'].iloc[i]
                    systemWithMaxCAR['safef'].iloc[i] = DPS[dps]['safef'].iloc[i]
                    systemWithMaxCAR['CAR25'].iloc[i] = DPS[dps]['CAR25'].iloc[i]
                    if metric != 'CAR25':
                        systemWithMaxCAR[metric].iloc[i] = DPS[dps][metric].iloc[i]
                    systemWithMaxCAR['dd95'].iloc[i] = DPS[dps]['dd95'].iloc[i]
                    systemWithMaxCAR['ddTol'].iloc[i] = DPS[dps]['ddTol'].iloc[i]
                #sMaxCAR.iloc[i] = maxM
        RS['RegimeSwitching_all_wl_'+metric] = systemWithMaxCAR
else:
    #only one wl
    wl = [key for key in dpsRunsByWL][0]
    for metric in metrics:
        print 'Creating Regime Switching using',metric
        if metric != 'CAR25':
            systemWithMaxCAR = pd.DataFrame(columns=['system','signals','gainAhead','safef','CAR25',\
                       metric,'dd95','ddTol'], index =DPS[dps].index)
        else:
            systemWithMaxCAR = pd.DataFrame(columns=['system','signals','gainAhead','safef','CAR25',\
                       'dd95','ddTol'], index =DPS[dps].index)
        for i in range(0,DPS[dps].shape[0]):
            #init
            maxM = rs_threshold
            systemWithMaxCAR['system'].iloc[i] = metric,'less than threshold of',rs_threshold
            systemWithMaxCAR['signals'].iloc[i] = 0
            systemWithMaxCAR['gainAhead'].iloc[i] = DPS[dps]['gainAhead'].iloc[i]
            systemWithMaxCAR['safef'].iloc[i] = 0
            systemWithMaxCAR['CAR25'].iloc[i] = 0
            if metric != 'CAR25':
                systemWithMaxCAR[metric].iloc[i] = 0
            systemWithMaxCAR['dd95'].iloc[i] = 0
            systemWithMaxCAR['ddTol'].iloc[i] = 0
            for dps in DPS:
                if DPS[dps][metric].iloc[i] > maxM:
                    maxM = DPS[dps][metric].iloc[i]
                    systemWithMaxCAR['system'].iloc[i] = dps
                    systemWithMaxCAR['signals'].iloc[i] = DPS[dps]['signals'].iloc[i]
                    systemWithMaxCAR['gainAhead'].iloc[i] = DPS[dps]['gainAhead'].iloc[i]
                    systemWithMaxCAR['safef'].iloc[i] = DPS[dps]['safef'].iloc[i]
                    systemWithMaxCAR['CAR25'].iloc[i] = DPS[dps]['CAR25'].iloc[i]
                    if metric != 'CAR25':
                        systemWithMaxCAR[metric].iloc[i] = DPS[dps][metric].iloc[i]
                    systemWithMaxCAR['dd95'].iloc[i] = DPS[dps]['dd95'].iloc[i]
                    systemWithMaxCAR['ddTol'].iloc[i] = DPS[dps]['ddTol'].iloc[i]
                #sMaxCAR.iloc[i] = maxM
        RS['RegimeSwitching_wl'+str(wl)+'_'+metric] = systemWithMaxCAR

#change back -1's and 1's to zeros
for dps in DPS:
    #print dps
    if string.split(dps,'__')[1][0] == 's':
        DPS[dps].signals[DPS[dps].signals == 1] = 0
    elif string.split(dps,'__')[1][0] == 'l':
        DPS[dps].signals[DPS[dps].signals == -1] = 0
    else:
        pass
        
#calc dps for RS 
RS_DPS = {}
for rs in RS:
    print 'Calculating DPS for', rs
    #RScurves[rs] = []
    sst_df = RS[rs][['system','signals','gainAhead']]
    sst_df = sst_df.set_index(pd.DatetimeIndex(sst_df.index))
    #redo DPS
    for wl in dpsRunsByWL:
        #append zeros for short window length since we dont have that data loaded
        zerobegin = pd.DataFrame(data=0, columns= ['system','signals','gainAhead'],\
            index = [sst_df.index[0] - datetime.timedelta(days=x) for x in range(int(math.ceil(wl)),0,-1)])
        sst_df = pd.concat([zerobegin, sst_df], axis=0)
        
        #dpsRun, sst_save = calcDPS2(rs, sst_df, PRT, start, end, wl, 'long')
        #RSofRS[dpsRun] = sst_save
        #RScurves[rs].append(dpsRun)
        
        #dpsRun, sst_save = calcDPS2(rs, sst_df, PRT, start, end, wl, 'short')
        #RSofRS[dpsRun] = sst_save
        #RScurves[rs].append(dpsRun)
        
        dpsRun, sst_save = calcDPS2(rs, sst_df, PRT, start, end, wl, 'both')
        if string.find(rs,'EMA') == -1:
            #add CAR25 series to df
            c25 = RS[rs].CAR25
            c25.name = 'C25'
            RS_DPS[dpsRun] = pd.concat([sst_save,c25],axis=1)
        else:
            #add EMA series to df
            RS_DPS[dpsRun] = pd.concat([sst_save,RS[rs].drop(['system', 'signals', \
                        'gainAhead','safef','CAR25','dd95','ddTol'], axis=1)],axis=1)
            
        #RScurves[rs].append(dpsRun)

#create equity Curves  
equityCurves = {}
for dps in RS:
    print 'creating equity curve for', dps
    equityCurves[dps] = calcEquity2(RS[dps], PRT['initial_equity'],dps[0])
    
for dps in RS_DPS:
    print 'creating equity curve for', dps
    equityCurves[dps] = calcEquity2(RS_DPS[dps], PRT['initial_equity'],dps[0])

for dps in DPS:
    print 'creating equity curve for ', dps
    equityCurves[dps] = calcEquity2(DPS[dps], PRT['initial_equity'],dps[0])

#create equity curve stats    
equityStats = pd.DataFrame(columns=['system','CAR25','CAR50','CAR75','DD100',\
                        'safef','cumCAR','MAXDD','sortinoRatio',\
                       'sharpeRatio','marRatio','k_ratio'], index = range(0,len(equityCurves)))


i=0
for sst in equityCurves:
    startDate = equityCurves[sst].index[0]
    endDate = equityCurves[sst].index[-1]
    years_in_forecast = (endDate-startDate).days/365.0
    #avgSafef = equityCurves[sst].safef.mean()
    
    #ALIGNMENT CHECK
    #pd.concat([close.ix[startDate:],equityCurves[sst].gainAhead], axis=1, join='outer').reset_index()
    close_aligned = pd.concat([close.ix[startDate:],equityCurves[sst].gainAhead], axis=1, join='outer').reset_index().Close
    CAR25_metrics = CAR25_df(sst, equityCurves[sst].signals, equityCurves[sst].reset_index().index, close_aligned,250)
    
    cumCAR = 100*(((equityCurves[sst].equity.iloc[-1]/equityCurves[sst].equity.iloc[0])**(1.0/years_in_forecast))-1.0) 
    MAXDD = max(equityCurves[sst].maxDD)*-100.0
    sortinoRatio = ratio(equityCurves[sst].equity).sortino()
    sharpeRatio = ratio(equityCurves[sst].equity).sharpe()
    marRatio = cumCAR/-MAXDD
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(equityCurves[sst].equity.values)),equityCurves[sst].equity.values)
    k_ratio =(slope/std_err) * math.sqrt(252.0)/len(equityCurves[sst].equity.values)
    
    equityStats.iloc[i].system = sst
    equityStats.iloc[i].CAR25 = CAR25_metrics['CAR25']
    equityStats.iloc[i].CAR50 = CAR25_metrics['CAR50']
    equityStats.iloc[i].CAR75 = CAR25_metrics['CAR75']
    equityStats.iloc[i].DD100 = -CAR25_metrics['DD100']
    equityStats.iloc[i].safef = CAR25_metrics['safef']
    #equityStats.iloc[i].avgSafef = avgSafef
    equityStats.iloc[i].cumCAR = cumCAR
    equityStats.iloc[i].MAXDD = MAXDD
    equityStats.iloc[i].sortinoRatio = sortinoRatio
    equityStats.iloc[i].sharpeRatio = sharpeRatio
    equityStats.iloc[i].marRatio = marRatio
    equityStats.iloc[i].k_ratio = k_ratio
    i+=1

#fill nan to zeros. happens when short system had no short signals
equityStats = equityStats.fillna(0)
#rank the curves based on scoring
#equityStats['avgSafefmm'] =minmax_scale(robust_scale(equityStats.avgSafef.reshape(-1, 1)))
equityStats['CAR25mm'] =minmax_scale(robust_scale(equityStats.CAR25.reshape(-1, 1)))
equityStats['CAR50mm'] =minmax_scale(robust_scale(equityStats.CAR50.reshape(-1, 1)))
equityStats['CAR75mm'] =minmax_scale(robust_scale(equityStats.CAR75.reshape(-1, 1)))
equityStats['DD100mm'] =minmax_scale(robust_scale(equityStats.DD100.reshape(-1, 1)))
equityStats['safefmm'] =minmax_scale(robust_scale(equityStats.safef.reshape(-1, 1)))
equityStats['cumCARmm'] =minmax_scale(robust_scale(equityStats.cumCAR.reshape(-1, 1)))
equityStats['MAXDDmm'] =minmax_scale(robust_scale(equityStats.MAXDD.reshape(-1, 1)))
equityStats['sortinoRatiomm'] = minmax_scale(robust_scale(equityStats.sortinoRatio.reshape(-1, 1)))
equityStats['marRatiomm'] =minmax_scale(robust_scale(equityStats.marRatio.reshape(-1, 1)))
equityStats['sharpeRatiomm'] =minmax_scale(robust_scale(equityStats.sharpeRatio.reshape(-1, 1)))
equityStats['k_ratiomm'] =minmax_scale(robust_scale(equityStats.k_ratio.reshape(-1, 1)))

equityStats['scoremm'] =  equityStats.CAR25mm+\
                        equityStats.DD100mm+equityStats.safefmm+equityStats.cumCARmm+equityStats.MAXDDmm+\
                        equityStats.sortinoRatiomm+equityStats.k_ratiomm+\
                        equityStats.sharpeRatiomm+equityStats.marRatiomm+\
                        equityStats.CAR50mm+equityStats.CAR75mm
                               
equityStats = equityStats.sort_values(['scoremm'], ascending=False)

estats_filename = systemName + startDate.strftime("%Y%m%d") +'to'+endDate.strftime("%Y%m%d")+'.csv'
equityStats.to_csv('/media/sf_Python/eStats_RS_'+estats_filename)

'''
topSystems = equityStats.system.iloc[0:nTopSystems]
#add top RS
if not sum(topSystems.str.contains('Regime')):
    topSystems = topSystems.tolist() + equityStats[equityStats.system.str.contains('Regime')].iloc[0:numRSCharts].system.values.tolist()
'''

#find top system
topSystem = equityStats.system.iloc[0]
#create benchmarks
benchmarks = createBenchmark(equityCurves[topSystem],1.0,'l', startDate,endDate,ticker)
benchmarks[topSystem] = equityCurves[topSystem]
#create yearly stats for benchmark
benchStatsByYear = createYearlyStats(benchmarks)
#create yearly stats for all equity curves with comparison against benchmark
equityCurvesStatsByYear = createYearlyStats(equityCurves, benchStatsByYear)

#display ranked charts
displayRankedCharts(1,benchmarks,benchStatsByYear,equityCurves,equityStats,equityCurvesStatsByYear)
displayRankedCharts(equityStats.system.shape[0],benchmarks,benchStatsByYear,equityCurves,equityStats,equityCurvesStatsByYear)

for i,sst in enumerate(equityStats.system.values): 
    equityCurves[sst].to_csv(rsSavePath+str(i)+'_'+systemName+'_'+sst+'.csv')
    
#for RS check alignment and analysis
RS_Analytics = {}
for r in RS_DPS:
    if string.find(r,'all') > 0:
        #for all window lengths
        c25Check = pd.DataFrame(index =DPS[dps].index)
        for dps in DPS:
            #is_ = string.split(dps[string.find(dps,'_is'):],'oos')[0]
            #algo = string.split(dps,'__')[1][:8]
            #wl_ = string.split(dps[string.find(dps,'wl'):])[0]
            if string.find(r,'CAR25') == -1:
                ema_name = r[string.find(r,'EMA'):].split()[0]
                ema = DPS[dps][ema_name]
                ema.name = ema_name + dps
                #ema.name = is_+algo+wl_ + '_' + ema_name
            else:
                ema_name = r[string.find(r,'CAR25'):].split()[0]
                ema = DPS[dps][ema_name]
                ema.name = ema_name + dps
                #ema.name = is_+algo+wl_ + '_' + ema_name
            #print c25.name
            c25 = DPS[dps].signals*DPS[dps].safef*DPS[dps].gainAhead
            c25.name = '_gA' #is_+algo+wl_ + '_gA'
            c25Check = pd.concat([c25Check, ema],axis=1)
            c25Check = pd.concat([c25Check, c25],axis=1)
        c25Check = pd.concat([RS_DPS[r],c25Check],axis=1)
    else:
        # by window length
        wl = float(string.split(r[string.find(r,'wl'):],'_')[0][2:])
        for dps in dpsRunsByWL[wl]:
            c25Check = pd.DataFrame(index =DPS[dps].index)
            #for dps in dpsRunsByWL[10]:
            for dps in DPS:
                #is_ = string.split(dps[string.find(dps,'_is'):],'oos')[0]
                #algo = string.split(dps,'__')[1][:8]
                #wl_ = string.split(dps[string.find(dps,'wl'):])[0]
                if string.find(r,'CAR25') == -1:
                    ema_name = r[string.find(r,'EMA'):].split()[0]
                    ema = DPS[dps][ema_name]
                    ema.name = ema_name + dps
                    #ema.name = is_+algo+wl_ + '_' + ema_name
                else:
                    ema_name = r[string.find(r,'CAR25'):].split()[0]
                    ema = DPS[dps][ema_name]
                    ema.name = ema_name + dps
                    #ema.name = is_+algo+wl_ + '_' + ema_name
                c25 = DPS[dps].signals*DPS[dps].safef*DPS[dps].gainAhead
                c25.name = '_gA' #is_+algo+wl_ + '_gA'
                c25Check = pd.concat([c25Check, ema],axis=1)
                c25Check = pd.concat([c25Check, c25],axis=1)
            c25Check = pd.concat([RS_DPS[r],c25Check],axis=1)
    RS_Analytics[r] = c25Check
#c25Check.to_csv('/media/sf_Python/c25check.csv')

for r in RS_Analytics:
    RS_Analytics[r].to_csv('/media/sf_Python/c25check_signals_sorted_all_labels'+r+'.csv')
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
'''
    #sort by is length
    colList1 = []
    colList2 = []
    for i,col in enumerate(RS_Analytics[r]):
        if col[:3] == '_is':
            #print col,
            ids = string.split(RS_Analytics[r].columns[i],'_')
            ids[1] = 'is'+'0'*(4-len(ids[1][2:]))+ids[1][2:]
            s = ''
            for x in ids:
                s += x+'_'
            #print s
            #is_ = string.split(col[string.find(col,'_is'):],'oos')[0]
            #algo = string.split(col,'__')[1][:8]
            #wl = string.split(col[string.find(col,'wl'):])[0]
            #c25Check[col].name = is_+algo+wl
            colList2.append(s)
        else:
            colList1.append(col)
    RS_Analytics[r].columns = colList1+colList2
    colList2.sort()
    
    c25Check_sorted = pd.DataFrame(data=RS_Analytics[r][colList1], index =RS_Analytics[r].index)
    for col in colList2:
        c25Check_sorted = pd.concat([c25Check_sorted, RS_Analytics[r][col]],axis=1)
    
    c25Check_sorted.to_csv('/media/sf_Python/c25check_signals_sorted_all_labels'+r+'.csv')
'''
