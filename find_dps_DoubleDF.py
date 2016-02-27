
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Wed Dec  2 06:10:48 2015

@author: hidemi

quick and dirty impementation

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
from datetime import datetime as dt
from pandas.core import datetools
import time
from suztoolz.transform import ratio
from suztoolz.loops import calcDPS2, calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.display import displayRankedCharts
from sklearn.preprocessing import scale, robust_scale, minmax_scale
import warnings
warnings.filterwarnings('error')   

start_time = time.time()
#  Set the path for the csv file
mypath = '/media/sf_Python/data/from_vf_to_dps/'
#mypath = '/media/sf_Python/data/from_df_to_dps2/'
dpsSavePath = '/media/sf_Python/data/from_DPS_to_RS/'
#filename = 'SST_F_ES_vf_VotingHard df_KNeighborsRegressor-distance_is250oos1_2014-01-02to2015-12-31_20160113074238.csv'
signal_type = 'ALL'
#metric to use for regime switching
metric = 'CAR25'
#numCharts = 2 # number charts to show
start = '2006-03-10'
end =  '2017-01-01'
#calcLongAndShort = True
shortWindowLength = [10,20]
longWindowLength = [2]
bothWindowLength = [2]
CAR25_threshold = 0
CAR25_threshold_short=-np.inf
#shortWindowLength = [5,10]
#longWindowLength = [5,10]
#bothWindowLength = [20]

PRT={}
PRT['DD95_limit'] = 0.20
PRT['tailRiskPct'] = 95
PRT['initial_equity'] = 1.0
PRT['horizon'] = 250
PRT['maxLeverage'] = 2

PRT_short={}
PRT_short['DD95_limit'] = 0.20
PRT_short['tailRiskPct'] = 95
PRT_short['initial_equity'] = 1.0
PRT_short['horizon'] = 250
PRT_short['maxLeverage'] = 1
#PRT['CAR25_threshold'] = -20 #set safef to 0 if < car threshold


#the files in the folder all need to be from the same time period
print 'Using',mypath,'to load files..'
aux_files = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
print 'loading', len(aux_files), 'files..'
systems = {}
for i,f in enumerate(aux_files):
#    if f != filename:   
        print f
        add_series = pd.read_csv(mypath+f, parse_dates={'dates':['Unnamed: 0']}, index_col='dates')
        #print add_series['systems'].shape
        systems[f[4:f.find('to')+12]] = add_series
ticker = f[f.find('F_'):f.find('F_')+4]
systemName = f[f.find('SST_')+4:f.find('SST_')+10]
#add date index
#signals = signals.set_index(pd.DatetimeIndex(add_series['Unnamed: 0'].values))
#gain ahead should all be the same for each file loaded
#gainAhead = pd.Series(data=add_series.gainAhead.values, index=signals.index, name='gainAhead')


#calc DPS
DPS = {}


#if calcLongAndShort == True:
for s in systems:
    DPS_short = {}
    DPS_long = {}
    DPS_both = {}
    startDate = systems[s].index[0]
    endDate = systems[s].index[-1]
    #print 'start',start, 'startdate',startDate, 'enddate',endDate
    for wl in shortWindowLength:
        sst_df_short = pd.concat([pd.Series(data=np.where(systems[s].signals==-1,-1,0), name='signals',\
                                    index=systems[s].index), systems[s].gainAhead], axis=1)
        sst_zero = pd.concat([pd.Series(data=0, name='signals', index=systems[s].index ), systems[s].gainAhead,\
                            pd.Series(data=0, name='safef', index=systems[s].index),pd.Series(data=0, name='CAR25', index=systems[s].index),\
                            pd.Series(data=0, name='dd95', index=systems[s].index),pd.Series(data=0, name='ddTol', index=systems[s].index)], axis=1)
        newStart = start
        sst_df_short = sst_df_short[sst_df_short.signals ==-1]
        #adjust starting date
        if sst_df_short.ix[:start].shape[0] < wl:
            #zerobegin = pd.DataFrame(data=0, columns= ['signals','gainAhead'],\
            #    index = [sst_df_short.index[0] - datetime.timedelta(days=x) for x in range(int(math.ceil(wl)),0,-1)])
            newStart = datetime.datetime.strftime(sst_df_short.iloc[wl].name,'%Y-%m-%d')
            print 'short DPS', start,'<=',startDate.date(),'adjusting', start,'to', newStart
            #sst_df_short = pd.concat([zerobegin, sst_df_short], axis=0)        
        dpsRun, sst_save_short = calcDPS2(s, sst_df_short, PRT, newStart, end, wl, trade='short', threshold=CAR25_threshold_short)
        #change long signals to 0
        DPS_short[dpsRun] = pd.concat([sst_save_short,\
                sst_zero.ix[[x for x in sst_zero.index if x not in sst_save_short.index]]],\
                axis=0,join='outer').sort_index().ix[newStart:]
                
    bestShortdpsRunName, bestShortDPS = findBestDPS(DPS_short, PRT_short, systems[s], start, end,'short', s)
    DPS[bestShortdpsRunName] = bestShortDPS


    #long DPS
    #print 'start',start, 'startdate',startDate, 'enddate',endDate
    for wl in longWindowLength:
        sst_df_long = pd.concat([pd.Series(data=np.where(systems[s].signals==1,1,0), name='signals',\
                                index=systems[s].index), systems[s].gainAhead], axis=1)
        sst_zero = pd.concat([pd.Series(data=0, name='signals', index=systems[s].index ), systems[s].gainAhead,\
                            pd.Series(data=0, name='safef', index=systems[s].index),pd.Series(data=0, name='CAR25', index=systems[s].index),\
                            pd.Series(data=0, name='dd95', index=systems[s].index),pd.Series(data=0, name='ddTol', index=systems[s].index)], axis=1)
        newStart = start
        sst_df_long = sst_df_long[sst_df_long.signals ==1]
        #adjust starting date
        if sst_df_long.ix[:start].shape[0] < wl:
            #zerobegin = pd.DataFrame(data=0, columns= ['signals','gainAhead'],\
            #    index = [sst_df_long.index[0] - datetime.timedelta(days=x) for x in range(int(math.ceil(wl)),0,-1)])
            newStart = datetime.datetime.strftime(sst_df_long.iloc[wl].name,'%Y-%m-%d')
            print 'long DPS', start,'<=',startDate.date(),'adjusting', start,'to', newStart
            #sst_df_long = pd.concat([zerobegin, sst_df_long], axis=0)            
        dpsRun, sst_save_long = calcDPS2(s, sst_df_long, PRT, newStart, end, wl, 'long', threshold=CAR25_threshold)
        #change short signals to 0
        DPS_long[dpsRun] = pd.concat([sst_save_long,\
                sst_zero.ix[[x for x in sst_zero.index if x not in sst_save_long.index]]],\
                axis=0,join='outer').sort_index().ix[newStart:]
                
    bestLongdpsRunName, bestLongDPS = findBestDPS(DPS_long, PRT, systems[s], start, end,'long', s)
    DPS[bestLongdpsRunName] = bestLongDPS

    #combine best long and short dps
    for i,dps in enumerate([bestShortDPS, bestLongDPS]):
        if i == 0:
            startDate = dps.index[0]
            endDate = dps.index[-1]
        else:
            if dps.index[0]>startDate:
                startDate =  dps.index[0]
            if  dps.index[-1]<endDate:
                endDate =  dps.index[-1]
    dpsRun = 'combined'
    if bestShortdpsRunName.find('wl') == -1:
        dpsRun += '_ShortNoDPS_'
    else:
        dpsRun += '_Short'+bestShortdpsRunName[bestShortdpsRunName.find('wl'):bestShortdpsRunName.find(' max')]

    if bestLongdpsRunName.find('wl') == -1:
        dpsRun += '_LongNoDPS_'
    else:
        dpsRun += '_Long'+bestLongdpsRunName[bestLongdpsRunName.find('wl'):bestLongdpsRunName.find(' max')]
        
    dpsRun += s
    DPS[dpsRun] = pd.concat([bestShortDPS[bestShortDPS.signals != 0].ix[startDate:],\
                bestLongDPS[bestLongDPS.signals != 0].ix[startDate:]], axis=0 ).sort_index()

    #print 'start',start, 'startdate',startDate, 'enddate',endDate
    for wl in bothWindowLength:
        sst_df = pd.concat([pd.Series(data=systems[s].signals, name='signals', index=systems[s].index),\
                            systems[s].gainAhead], axis=1)

        newStart = start            
        if sst_df.ix[:start].shape[0] < wl:
            #zerobegin = pd.DataFrame(data=0, columns= ['signals','gainAhead'],\
            #    index = [sst_df.index[0] - datetime.timedelta(days=x) for x in range(int(math.ceil(wl)),0,-1)])
            newStart = datetime.datetime.strftime(sst_df.iloc[wl].name,'%Y-%m-%d')
            print 'both DPS', start,'<=',startDate.date(),'adjusting', start,'to', newStart
            #sst_df = pd.concat([zerobegin, sst_df], axis=0)
            
        dpsRun, sst_save = calcDPS2(s, sst_df, PRT, newStart, end, wl, 'both', threshold=CAR25_threshold)
        DPS_both[dpsRun] = sst_save
        
        #dpsRun, sst_save = calcDPS2('BuyHold', buyandhold, PRT, start, end, wl)
        #DPS[dpsRun] = sst_save
        
    dpsRunName, bestBothDPS = findBestDPS(DPS_both, PRT, systems[s], start, end,'both', s)
    DPS[dpsRunName] = bestBothDPS




#  compute equity, maximum equity, drawdown, and maximum drawdown
equityCurves = {}
DPS_adj = {}
#set the start/end dates to ones that all equity curves share for apples to apples comparison.
startDate = dt.strptime(start,'%Y-%m-%d')
endDate = dt.strptime(end,'%Y-%m-%d')

for dps in DPS:
    if DPS[dps].index[0]>startDate:
        startDate =  DPS[dps].index[0]
    if  DPS[dps].index[-1]<endDate:
        endDate =  DPS[dps].index[-1]
        
DPS_adj = {}
for dps in DPS:
    DPS_adj[dps] = DPS[dps].ix[startDate:]
    
for sst in DPS_adj:
    print 'creating equity curve for ', sst
    equityCurves[sst] = calcEquity2(DPS_adj[sst], PRT['initial_equity'],sst[0])
    #print DPS_adj[sst].tail(60)
    
    #DPS_adj[sst] = DPS_adj[sst].set_index(pd.DatetimeIndex(DPS_adj[sst]['index']))
    #DPS_adj[sst] = DPS_adj[sst].drop('index', axis=1)
'''
#create equity curves for safef1
for sst in systems:
    print 'creating equity curve for ', sst
    systems[sst] = systems[sst].ix[startDate:]
    system_sst = pd.concat([systems[sst],\
                        pd.Series(data=1.0, name = 'safef', index = systems[sst].index),
                        pd.Series(data=np.nan, name = 'CAR25', index = systems[sst].index),
                        pd.Series(data=np.nan, name = 'dd95', index = systems[sst].index),
                        ],axis=1)
    equityCurves[sst] = calcEquity2(system_sst, PRT['initial_equity'],'b')
'''        
    
#create equity curve stats    
equityStats = pd.DataFrame(columns=['system','avgSafef','cumCAR','MAXDD','sortinoRatio',\
                       'sharpeRatio','marRatio','k_ratio'], index = range(0,len(equityCurves)))
#this calc dosent exclude non-trading days
years_in_forecast = (endDate-startDate).days/365.0
i=0
for sst in equityCurves:
    avgSafef = equityCurves[sst].safef.mean()    
    cumCAR = 100*(((equityCurves[sst].equity.iloc[-1]/equityCurves[sst].equity.iloc[0])**(1.0/years_in_forecast))-1.0) 
    MAXDD = max(equityCurves[sst].maxDD)*-100.0
    sortinoRatio = ratio(equityCurves[sst].equity).sortino()
    sharpeRatio = ratio(equityCurves[sst].equity).sharpe()
    marRatio = cumCAR/-MAXDD
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(equityCurves[sst].equity.values)),equityCurves[sst].equity.values)
    k_ratio =(slope/std_err) * math.sqrt(252.0)/len(equityCurves[sst].equity.values)
    
    equityStats.iloc[i].system = sst
    equityStats.iloc[i].avgSafef = avgSafef
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
equityStats['avgSafefmm'] =minmax_scale(robust_scale(equityStats.avgSafef.reshape(-1, 1)))
equityStats['cumCARmm'] =minmax_scale(robust_scale(equityStats.cumCAR.reshape(-1, 1)))
equityStats['MAXDDmm'] =minmax_scale(robust_scale(equityStats.MAXDD.reshape(-1, 1)))
equityStats['sortinoRatiomm'] = minmax_scale(robust_scale(equityStats.sortinoRatio.reshape(-1, 1)))
equityStats['marRatiomm'] =minmax_scale(robust_scale(equityStats.marRatio.reshape(-1, 1)))
equityStats['sharpeRatiomm'] =minmax_scale(robust_scale(equityStats.sharpeRatio.reshape(-1, 1)))
equityStats['k_ratiomm'] =minmax_scale(robust_scale(equityStats.k_ratio.reshape(-1, 1)))

equityStats['scoremm'] =  equityStats.avgSafefmm+equityStats.cumCARmm+equityStats.MAXDDmm+\
                                equityStats.sortinoRatiomm+equityStats.k_ratiomm+\
                               equityStats.sharpeRatiomm+equityStats.marRatiomm

equityStats = equityStats.sort_values(['scoremm'], ascending=False)


        
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
'''
# chart by system: non-dps and dps
#create a list of systems ranked by their DPS_adj result
dps_ranked_systems = []
for dps in [x for x in equityStats.system if x in DPS_adj]:
    for sst in systems:
        if string.find(dps,sst) !=-1 and sst not in dps_ranked_systems:
            dps_ranked_systems.append(sst)
            
#chart them one by one
if len(DPS_adj) == len(dps_ranked_systems):
    #case for one windowlength for each system.
    for i,sst in enumerate(dps_ranked_systems):
        benchmarks = createBenchmark(equityCurves[topSystem],1.0,'l', startDate,endDate,ticker)
        benchmarks[sst] = equityCurves[sst]
        for dps in DPS_adj:
            #print string.find(dps,sst)
            if string.find(dps,sst) !=-1:
                #stop at dps match
                break
        ec = {}
        ec[dps] = equityCurves[dps]
        ec[sst] = equityCurves[sst]
        es = equityStats[equityStats.system == dps]
        benchStatsByYear = createYearlyStats(benchmarks)
        equityCurvesStatsByYear = createYearlyStats(ec, benchStatsByYear)
        displayRankedCharts(1,benchmarks,benchStatsByYear,ec,es,equityCurvesStatsByYear, vsDPS=True, dpsRank=i, dpsChartRank=0)
else:
    #case for multiple windowlength for each system. 
    for i,sst in enumerate(dps_ranked_systems):
        #for each sytem display all chart for that system. 
        benchmarks = createBenchmark(equityCurves[topSystem],1.0,'l', startDate,endDate,ticker)
        benchmarks[sst] = equityCurves[sst]
        for j, dps in enumerate([x for x in equityStats.system if string.find(x,sst) !=-1]):
            #print string.find(dps,sst), dps
            ec = {}
            ec[dps] = equityCurves[dps]
            ec[sst] = equityCurves[sst]
            es = equityStats[equityStats.system == dps]
            benchStatsByYear = createYearlyStats(benchmarks)
            equityCurvesStatsByYear = createYearlyStats(ec, benchStatsByYear)
            displayRankedCharts(1,benchmarks,benchStatsByYear,ec,es,equityCurvesStatsByYear, vsDPS=True, dpsRank=i, dpsChartRank=j)
'''      
prt=''
for x in PRT:
    prt+= x+str(PRT[x])

filename = startDate.strftime("%Y%m%d") +'to'+endDate.strftime("%Y%m%d")+prt+'.csv'
equityStats.to_csv('/media/sf_Python/eStats_'+systemName+filename)

#save DPS_adj for RS
for dps in DPS_adj:
    print 'saving', dps, 'to', dpsSavePath
    DPS_adj[dps].to_csv(dpsSavePath+systemName+dps+'.csv', index_label = 'dates')
    
#for i,sst in enumerate(topSystems): 
#    equityCurves[sst].sort_index(ascending=False).to_csv('/media/sf_Python/data/from_RS/'+str(i)+'_'+ticker+sst+'.csv')
    
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
