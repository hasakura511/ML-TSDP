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
import time
import datetime
import matplotlib.dates as mdates
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
import math
from abc import ABCMeta, abstractmethod
import matplotlib.animation as animation

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

    def __init__(self):
        print 'Initializing MarketIntradayPortfolio'
        self.ready=False
        
    def setInit(self, symbol, bars, signals, nextSig, lastSig, initial_capital=100000.0, shares=500):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.nextSig=nextSig
        self.lastSig=lastSig
        self.initial_capital = float(initial_capital)
        self.shares = int(shares)
        self.positions = self.generate_positions()
        self.returns = dict()
        self.ready=False
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
            
        #portfolio['price_diff'] = self.bars['Close_Out']-self.bars['Open_Out']
        portfolio['price_diff'] = self.bars['Close']-self.bars['Open']
        #portfolio['price_diff'][0:5] = 0.0
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']
     
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        self.returns=portfolio
        self.ready=True
        return portfolio
        
    def plot_graph(self):
        # Plot results
        lines=list()
        f, ax = plt.subplots(3, sharex=True)
        f.patch.set_facecolor('white')
        while not self.ready:
            time.sleep(1)
        ylabel = self.symbol + ' Close Price in $'
        #self.bars['Close_Out'].plot(ax=ax[0], color='r', lw=3.)    
        line, =ax[0].plot(self.bars['Close'], color='r', lw=3.)   
        lines.append(line)
        ax[0].set_ylabel(ylabel, fontsize=8)
        ax[0].set_xlabel('', fontsize=18)
        ax[0].legend(('Close Price ' + self.symbol,), loc='upper left', prop={"size":8})
        ax[0].set_title('(' + str(self.bars.index[0]) + '-' + str(self.bars.index[-1]) +')')
        
        line, =ax[1].plot(self.returns['total'], color='b', lw=3.) 
        lines.append(line)
        ax[1].set_ylabel('Portfolio value in $', fontsize=8)
        ax[1].set_xlabel('Date', fontsize=18)
        ax[1].legend(('Portofolio Performance. Capital Invested: 100k $. Shares Traded per day: 500+500',), loc='upper left', prop={"size":8})            
        
        width=0.35
        line=ax[2].bar(np.array(self.bars.index)[np.array(self.signals['signal']) > 0], np.array(self.signals['signal'])[np.array(self.signals['signal']) > 0], color='g', edgecolor='none') 
        line=ax[2].bar(np.array(self.bars.index)[np.array(self.signals['signal']) < 0], np.array(self.signals['signal'])[np.array(self.signals['signal']) < 0], color='r', edgecolor='none') 
        lines.append(line)
        ax[2].set_ylabel('Signals', fontsize=8)
        ax[2].legend(('Signal',), loc='upper left', prop={"size":8})            
        
        cmt1=ax[1].annotate('Next Signal: ' + str(self.nextSig) + ' Last Close: ' + str(self.bars['Close'][-1]), 
                    xy=(0.02, -0.04), ha='left', va='top', xycoords='axes fraction', fontsize=10)
        
        plt.tick_params(axis='both', which='major', labelsize=8)
        
        loc = ax[1].xaxis.get_major_locator()
        loc.maxticks[DAILY] = 24
        def draw_arrow():
            r = 20  # or whatever fits you
            if self.returns['price_diff'][-1] > 0:
                ax[0].annotate('Chg: ' + str(round(self.returns['price_diff'][-1],2)) + '\nPL: $' + 
                    str(round(self.returns['profit'][-1])), xy=(self.bars.index[-1],self.bars['Close'][-1]),
                    xytext=(0, -55), textcoords='offset points',ha='center',fontsize=8,
                    arrowprops=dict(facecolor='green', shrink=0.1), backgroundcolor='white'
                    )
                   
            else:
                ax[0].annotate('Chg: ' + str(round(self.returns['price_diff'][-1],2)) + '\nPL: $' + 
                    str(round(self.returns['profit'][-1],2)), xy=(self.bars.index[-1],self.bars['Close'][-1]),
                    xytext=(0, 55), textcoords='offset points',ha='center',fontsize=8,
                    arrowprops=dict(facecolor='red', shrink=0.1),backgroundcolor='white'
                    )
            if self.nextSig > 0:
                ax[1].annotate('Next Signal', 
                    xy=(self.bars.index[-1],self.returns['total'][-1]),
                    xytext=(0, -25), textcoords='offset points', ha='center',fontsize=8,
                    arrowprops=dict(facecolor='green', shrink=0.1), backgroundcolor='white'
                    )
            else:
                ax[1].annotate('Next Signal', xy=(self.bars.index[-1],self.returns['total'][-1]),
                    xytext=(0, 25), textcoords='offset points',ha='center',fontsize=8,
                    arrowprops=dict(facecolor='red', shrink=0.1), backgroundcolor='white'
                    )
                    
            if (self.returns['price_diff'][-1] > 0 and self.lastSig > 0) or \
               (self.returns['price_diff'][-1] < 0 and self.lastSig < 0):
                coord=[0,-35]
                if self.lastSig > 0:
                    coord=[0,-35]
                else:
                    coord=[0,35]
                ax[1].annotate('O', 
                    xy=(self.bars.index[-2],self.returns['total'][-2]), 
                    xytext=coord, ha='center',  
                        textcoords='offset points', fontsize=20, color='green')
                        #arrowprops=dict(facecolor='green', shrink=0.001))
            elif self.lastSig != 0:
                coord=[0,-35]
                if self.lastSig > 0:
                    coord=[0,-35]
                else:
                    coord=[0,35]
                #xy=(self.bars.index[-2],self.returns['total'][-2])
                ax[1].annotate('X', 
                    xy=(self.bars.index[-2],self.returns['total'][-2]), 
                    xytext=coord, ha='center', 
                        textcoords='offset points', fontsize=20, color='red')
                        #arrowprops=dict(facecolor='red', shrink=0.001))
            
            
                    
        draw_arrow()  
        def update(*args):
            #lines[0].set_ydata(self.bars['Close'])
            #lines[1].set_ydata(self.returns['total'])
            line, =ax[0].plot(self.bars['Close'], color='r', lw=3.)   
            line, =ax[1].plot(self.returns['total'], color='b', lw=3.) 
            line=ax[2].bar(np.array(self.bars.index)[np.array(self.signals['signal']) > 0], np.array(self.signals['signal'])[np.array(self.signals['signal']) > 0], color='g', edgecolor='none') 
            line=ax[2].bar(np.array(self.bars.index)[np.array(self.signals['signal']) < 0], np.array(self.signals['signal'])[np.array(self.signals['signal']) < 0], color='r', edgecolor='none') 
            ax[0].set_title('(' + str(self.bars.index[0]) + '-' + str(self.bars.index[-1]) +')')
            f.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            f.autofmt_xdate()
            cmt1=ax[1].annotate(' Open: ' + str(self.bars['Open'][-1]) + ' Last Signal: ' + str(self.signals['signal'][-1]) + 
            ' Close: ' + str(self.bars['Close'][-1])  + ' Next Signal: ' + str(self.nextSig) + '  ', 
                    xy=(0.02, -0.04), ha='left', va='top', xycoords='axes fraction', fontsize=10, backgroundcolor='white')
            
            draw_arrow()
            #plt.draw()
            # Return modified artists
            return lines
        anim = animation.FuncAnimation(f, update, interval=2)
        
        plt.show()

