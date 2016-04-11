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
from matplotlib.dates import DAILY
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
import pandas as pd
import features
import classifier
import data

from abc import ABCMeta, abstractmethod

class Portfolio(object):
    """An abstract base class representing a portfolio of 
    positions (including both instruments and cash), determined
    on the basis of a set of signals provided by a Strategy."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        """Provides the logic to determine how the portfolio 
        positions are allocated on the basis of forecasting
        signals and available cash."""
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """Provides the logic to generate the trading orders
        and subsequent equity curve (i.e. growth of total equity),
        as a sum of holdings and cash, and the bar-period returns
        associated with this curve based on the 'positions' DataFrame.

        Produces a portfolio object that can be examined by 
        other classes/functions."""
        raise NotImplementedError("Should implement backtest_portfolio()!")


class MarketIntradayPortfolio(Portfolio):
    """Buys or sells 500 shares of an asset at the opening price of
    every bar, depending upon the direction of the forecast, closing 
    out the trade at the close of the bar.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=100000.0, shares=500):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.shares = int(shares)
        self.positions = self.generate_positions()
        self.returns = dict()
        
    def generate_positions(self):
        """Generate the positions DataFrame, based on the signals
        provided by the 'signals' DataFrame."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)

        positions[self.symbol] = self.shares*self.signals['signal']
        return positions
                    
    def backtest_portfolio(self):
        """Backtest the portfolio and return a DataFrame containing
        the equity curve and the percentage returns."""
       
        portfolio = pd.DataFrame(index=self.positions.index)
        self.pos_diff = self.positions.diff()
            
        portfolio['price_diff'] = self.bars['Close_Out']-self.bars['Open_Out']
        portfolio['price_diff'][0:5] = 0.0
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']
     
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        self.returns=portfolio
        return portfolio
        
    def plot_graph(self):
        # Plot results
        f, ax = plt.subplots(2, sharex=True)
        f.patch.set_facecolor('white')
        ylabel = self.symbol + ' Close Price in $'
        self.bars['Close_Out'].plot(ax=ax[0], color='r', lw=3.)    
        ax[0].set_ylabel(ylabel, fontsize=18)
        ax[0].set_xlabel('', fontsize=18)
        ax[0].legend(('Close Price S&P-500',), loc='upper left', prop={"size":8})
        ax[0].set_title('S&P 500 Close Price VS Portofolio Performance (' + str(self.bars.index[0]) + '-' + str(self.bars.index[-1]) +')')
        
        self.returns['total'].plot(ax=ax[1], color='b', lw=3.)  
        ax[1].set_ylabel('Portfolio value in $', fontsize=8)
        ax[1].set_xlabel('Date', fontsize=18)
        ax[1].legend(('Portofolio Performance. Capital Invested: 100k $. Shares Traded per day: 500+500',), loc='upper left', prop={"size":8})            
        plt.tick_params(axis='both', which='major', labelsize=8)
        loc = ax[1].xaxis.get_major_locator()
        loc.maxticks[DAILY] = 24
    
        #figManager = plt.get_current_fig_manager()
        #figManager.frame.Maximize(True)
        
        plt.show()

