# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 04:52:49 2016

@author: hidemi
"""
import numpy as np
import pandas as pd
import Quandl
dataLoadStartDate = "1998-12-22"
dataLoadEndDate = "2017-01-01"
print "\nReading data from Quandl... "
savepath= 'D:/Dropbox/SharedTSDP/data/quandl/'
tickers = ["YAHOO/INDEX_SSEC","YAHOO/INDEX_HSI","GOOG/NYSE_XLY","GOOG/NYSE_XLP",\
            "GOOG/NYSE_XLE","GOOG/NYSE_XLU","GOOG/NYSE_XLF",\
        "GOOG/NYSE_XLV","GOOG/NYSE_XLI","GOOG/NYSE_XLB","GOOG/NYSE_XLK"]

#tickers = ["GOOG/NYSE_XLY","GOOG/NYSE_XLP","GOOG/NYSE_XLE","GOOG/NYSE_XLU","GOOG/NYSE_XLF",\
#        "GOOG/NYSE_XLV","GOOG/NYSE_XLI","GOOG/NYSE_XLB","GOOG/NYSE_XLK",]
        #"GOOG/NYSE_XLFS","GOOG/NYSE_XLRE"]

for t in tickers:
    qt = Quandl.get(t, trim_start=dataLoadStartDate,
                        trim_end=dataLoadEndDate, 
                   authtoken="aiZ4bv-njY-g1GSuYsvJ")
    qt.to_csv(savepath+t[-3:]+'.csv')
    print 'saving', t,'as', t[-3:]+'.csv'
