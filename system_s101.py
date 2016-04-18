import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
from sklearn.qda import QDA
import re
from dateutil import parser

import datetime

import numpy as np
import matplotlib.pyplot as plt
try:
    from matplotlib.finance import quotes_historical_yahoo
except ImportError:
    from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

from os import listdir
from os.path import isfile, join
import re
import sys
import pandas as pd
import p.features
import p.classifier
import p.data
import p.backtest
import logging
import threading
import seitoolz.signal as signal
import p.model as model
import pytz
from tzlocal import get_localzone

logging.basicConfig(filename='/logs/system_s101.log',level=logging.DEBUG)

if len(sys.argv) > 3 and sys.argv[2] == '3':
    lookback=int(sys.argv[3])
    threads=list()
    sig_thread = threading.Thread(target=model.start_lookback, args=[lookback, sys.argv])
    sig_thread.daemon=True
    threads.append(sig_thread)
    [t.start() for t in threads]
    model.portfolio.plot_graph()
    #[t.join() for t in threads]
else:
     nextSignal=model.get_signal(0, model.portfolio, sys.argv)
     if (len(sys.argv) > 2 and sys.argv[2] == '2'):
         sysfile='s101'
         if sys.argv[1]=='1':
             sysfile='s101_ES'
         if sys.argv[2]=='2':
             sysfile='s101_EURJPY'
             if len(sys.argv) > 4:
                 sysfile='s101_' + sys.argv[4]
         if sys.argv[2]=='9':
             sysfile='s101v2_EURJPY'
             if len(sys.argv) > 4:
                 sysfile='s101v2_' + sys.argv[4]
         eastern=pytz.timezone('US/Eastern')
         nowDate=datetime.datetime.now(get_localzone()).astimezone(eastern)
         signal.generate_model_sig(sysfile, str(nowDate), nextSignal, 1, '')
         
     else:
         model.portfolio.plot_graph()