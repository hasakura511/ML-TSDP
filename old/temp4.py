# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

import numpy as np
import math
import talib as ta
import pandas as pd
from suztoolz.transform import zigzag as zg
import arch
from os import listdir
from os.path import isfile, join

from datetime import datetime
import matplotlib.pyplot as plt
#from pandas.io.dataSet import DataReader
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import time
from suztoolz.transform import perturb_data
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio

from suztoolz.loops import CAR25, CAR25_prospector, maxCAR25
from suztoolz.display import init_report, update_report_prospector,\
                            display_CAR25, compareEquity_vf, getToxCDF, adf_test,\
                            describeDistribution
from sklearn.grid_search import ParameterGrid
import re

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
import logging
import os
from pytz import timezone
from dateutil.parser import parse
#dataSet = pd.read_csv('./data/from_IB/30m_EURCHF.csv')
minDatapoints = 5
zz_std = 1
supportResistanceLB = 250
#for i in range(supportResistanceLB,dataSet.shape[0]):
for i in range(supportResistanceLB,500):
    data = dataSet[i-supportResistanceLB:i]
    data.index = data.index.to_datetime()
    zz = zg(data.Close,data.Close.std()*zz_std,-data.Close.std()*zz_std)
    
    data = dataSet[i-supportResistanceLB:i].reset_index()
    peaks = np.where(zz.peak_valley_pivots()==1)[0]
    if data.Close.idxmax() in peaks and supportResistanceLB-data.Close.idxmax()>minDatapoints:
        #choose local maximum
        lastPeak = data.Close.idxmax()
    else:
        #choose last max point
        lastPeak = [x for x in peaks if supportResistanceLB-x>minDatapoints][-1]
    
    valleys = np.where(zz.peak_valley_pivots()==-1)[0]
    if data.Close.idxmin() in valleys and supportResistanceLB-data.Close.idxmin()>minDatapoints:
        lastValley= data.Close.idxmin()
    else:
        lastValley = [x for x in valleys if supportResistanceLB-x>minDatapoints][-1]
    
    zz.plot_pivots(l=8,w=8,mv=(lastValley, data.Close[lastValley]), mp=(lastPeak, data.Close[lastPeak]))
    wfSteps = sorted([supportResistanceLB-lastPeak, supportResistanceLB-lastValley])
    print 'wfSteps', wfSteps, 'axis', lastPeak, lastValley
    
    #print i, 'high index',data.Close.idxmax(), 'high',data.Close[data.Close.idxmax()], max(data.Close)
    #print i, 'low index',data.Close.idxmin(), 'low',data.Close[data.Close.idxmin()], min(data.Close)
    #print supportResistanceLB-data.Close.idxmin(), supportResistanceLB-data.Close.idxmax()