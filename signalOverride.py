# -*- coding: utf-8 -*-
"""
Created on Sun Jul 03 11:26:48 2016

@author: Hidemi
"""

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
filename='system_override.csv'

if len(sys.argv) > 1:
    #bestParamsPath = './data/params/'
    signalPath = './data/signals/'
    dataPath = './data/systems/'
    #equityCurveSavePath = './data/signalPlots/'
    #savePath== './data/results/'
    #pngPath = './data/results/'
    #showPlot = False
    #verbose = False
else:
    #signalPath = 'D:/ML-TSDP/data/signals/' 
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/'
    dataPath = 'D:/ML-TSDP/data/systems/'
    #dataPath = 'D:/ML-TSDP/data/csidata/v4futures2/'
    #bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    #equityCurveSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #savePath= 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #pngPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #showPlot = True
    #verbose = True

sigOverride = pd.read_csv(dataPath+filename, index_col=0)
nsig=0
for ticker in sigOverride.index:
    nsig+=1
    signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
    #addLine = signalFile.iloc[-1]
    #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
    #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
    #addLine.name = sst.iloc[-1].name
    addLine = pd.Series(name=signalFile.index[-1])
    addLine['signals']=sigOverride.ix[ticker].signal
    addLine['safef']=sigOverride.ix[ticker].safef
    addLine['timestamp']=dt.now().strftime("%Y%m%d %H:%M:%S %Z")
    signalFile = signalFile.append(addLine)
    filename=signalPath + version+'_'+ ticker+ '.csv'
    print 'Saving...\n',addLine,  filename
    signalFile.to_csv(filename, index=True)
print nsig, 'files updated'
