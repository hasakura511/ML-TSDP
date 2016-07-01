import os
import math
import string
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pytz import timezone
from datetime import datetime as dt
from os import listdir
from os.path import isfile, join
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
import seaborn as sns

version = 'v4'
if len(sys.argv) > 1:
    #bestParamsPath = './data/params/'
    #signalPath = './data/signals/'
    #dataPath = './data/csidata/v4futures2/'
    equityCurveSavePath = './data/signalPlots/'
    savePath== './data/results/'
    #pngPath = './data/results/'
    #showPlot = False
    #verbose = False
else:
    #signalPath = 'D:/ML-TSDP/data/signals/' 
    #signalPath = 'D:/ML-TSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    #dataPath = 'D:/ML-TSDP/data/csidata/v4futures2/'
    #bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    equityCurveSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    savePath= 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #pngPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #showPlot = True
    #verbose = True

dataFiles = [ f for f in listdir(equityCurveSavePath) if \
                    isfile(join(equityCurveSavePath,f)) and '.csv' in f and version+'_' in f and\
                    len(f.split('_')[1].split('.')[0])<6]
print len(dataFiles), dataFiles

futuresRank=pd.DataFrame()
for f in dataFiles:
    sym = f.split('_')[1].split('.')[0]
    df = pd.read_csv(equityCurveSavePath+f, index_col=0)
    df.index.name='dates'
    #print df
    futuresRank.set_value(sym,df.index[-1],df.equity[-1])
futuresRank = futuresRank.sort_values(by=futuresRank.columns[0], ascending=False)
print 'Saving', savePath+'fRank.html'
futuresRank.to_html(savePath+'fRank.html')
print futuresRank