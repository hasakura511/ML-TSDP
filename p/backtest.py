import cPickle
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from random import randint
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
import pandas as pd
import datetime
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import proj3d
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
from matplotlib.dates import DAILY
import matplotlib as mpl
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
        
    def setInit(self, symbol, bars, signals, algos, nextSig, lastSig, parameters, initial_capital=0.0, shares=500):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.algos=algos
        self.nextSig=nextSig
        self.lastSig=lastSig
        self.initial_capital = float(initial_capital)
        self.shares = int(shares)
        self.positions = self.generate_positions()
        self.returns = dict()
        self.ready=False
        self.parameters=parameters
        
    def generate_positions(self):
        """Generate the positions DataFrame, based on the signals
        provided by the 'signals' DataFrame."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        for algo in self.algos:
            positions[algo] = self.shares*self.signals[algo]
        return positions
                    
    def backtest_portfolio(self):
        """Backtest the portfolio and return a DataFrame containing
        the equity curve and the percentage returns."""
        self.rank=dict()
        portfolio = dict()
        for algo in self.algos:
            portfolio[algo]=pd.DataFrame(index=self.positions.index)
            self.pos_diff=dict()
            self.pos_diff[algo] = self.positions[algo].diff()
                
            portfolio[algo]['price_diff'] = self.bars['Close']-self.bars['Open']
            #portfolio['price_diff'][0:5] = 0.0
            portfolio[algo]['profit'] = self.positions[algo] * portfolio[algo]['price_diff']
            portfolio[algo]['total'] = self.initial_capital + portfolio[algo]['profit'].cumsum()
            portfolio[algo]['returns'] = portfolio[algo]['total'].pct_change()
            self.rank[algo]=float(portfolio[algo]['total'][-1] - portfolio[algo]['total'][0])
            self.returns=portfolio
        #self.ranking= sorted(self.rank.items(), key=operator.itemgetter(1), reverse=True)
        self.ranking= sorted(self.rank.items(), key=operator.itemgetter(1))
        self.ready=True
        return (portfolio, self.rank, self.ranking)
        
    def plot_graph(self):
        # Plot results
        #f, ax = plt.subplots(2, sharex=True)
        while not self.ready:
            time.sleep(1)
        
        ylabel = self.symbol + ' Close Price in $'
        fig, [ax1, ax4, ax2, ax3] = plt.subplots(4, 1, sharey=True, figsize=(10, 8), subplot_kw={'projection': '3d'})  
        fig.patch.set_facecolor('white')
        lines1=list()
        color2=dict()
        def draw_close():
            count = len(self.ranking)
            for line in lines1:
                line.remove()
            del lines1[:]
            for [algo,equity] in self.ranking:
                #if not color2.has_key(algo):
                color2[algo]='#'+'%06X' % randint(0, 0xFFFFFF)
                #ax1.plot_wireframe(np.array(self.bars.index), np.array(self.bars['Close']), count, rstride=11, cstride=0)
                line=ax1.plot_wireframe(count, mpl.dates.date2num(self.bars.index.to_pydatetime()), 
                                        np.array(self.bars['Close']), rstride=10, cstride=10, 
                                        color=color2[algo], label=str(count)+' ' + algo)
                #line, =ax1.plot(self.bars['Close'], color='r', lw=3.)   
                lines1.append(line)
                
                
                count = count - 1
                #, 
                              #size=10,zorder=1)    
            ax1.set_zlabel(ylabel, fontsize=8)
            ax1.legend(loc='upper left', fontsize=8)
            
        lines2=list()
        #global maxequity
        #maxequity=0
        def draw_equity():
            #global maxequity
            for line in lines2:
                    line.remove()
            del lines2[:]
            count = len(self.ranking)
            for [algo,equity] in self.ranking:
                if not color2.has_key(algo):
                    color2[algo]='#'+'%06X' % randint(0, 0xFFFFFF)
                    
                line=ax2.plot_wireframe(count, mpl.dates.date2num(self.bars.index.to_pydatetime()),
                                        np.array(self.returns[algo]['total'], dtype=float), 
                                        rstride=10, cstride=10, color=color2[algo], label=algo)
                lines2.append(line)
                #if self.returns[algo]['total'][-1] > maxequity:
                #    maxequity=self.returns[algo]['total'][-1]
                #cmt1=ax2.text(mpl.dates.date2num(self.bars.index.to_pydatetime())[0], count,
                #              0, 
                #              ' Return: $' + str(equity),
                #              fontsize=5)
                #lines2.append(cmt1)
                
                ydepth=np.empty(len(self.bars.index))
                ydepth.fill(count)
                pg=np.array(self.returns[algo]['total'])
                colordepth=np.empty(len(self.bars.index)).astype(str)
                colordepth.fill('green')
                colordepth[np.array(self.returns[algo]['profit']) < 0]='red'
                zpos=np.empty(len(self.bars.index))
                zpos.fill(0)
                line=ax2.bar3d(ydepth, mpl.dates.date2num(self.bars.index.to_pydatetime()),
                     zpos, 
                       0.5,0.5,np.array(pg),
                   color=colordepth, zsort='average', alpha=0.5, edgecolor='none')
                lines2.append(line)
                
                count = count - 1
            ax2.set_zlabel('Portfolio value in $', fontsize=8)
            green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
            red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
            ax2.legend([green_proxy,red_proxy],['Signal Right','Signal Wrong'], loc='upper left', fontsize=8) 
            #ax2.set_ylabel('Date', fontsize=8)
            #ax2.legend(loc='upper left', fontsize=8)            

        #fig = plt.figure()
        #ax3 = fig.add_subplot(111, projection='3d')
        lines3=list()
        def draw_signals():
            width=-0.35
        
            for line in lines3:
                    line.remove()
            del lines3[:]
            count = len(self.ranking)
            for [algo,equity] in self.ranking:
                if not color2.has_key(algo+'2'):
                    color2[algo+'2']='#'+'%06X' % randint(0, 0xFFFFFF)
                #ydepth=np.empty(len(mpl.dates.date2num(self.bars.index.to_pydatetime())[np.array(self.signals[algo]) > 0]))
                #ydepth.fill(count)
                #line=ax3.bar3d(mpl.dates.date2num(self.bars.index.to_pydatetime())[np.array(self.signals[algo]) > 0], 
                #     ydepth, np.array(self.signals[algo])[np.array(self.signals[algo]) > 0],
                #     1,1,1,
                #     color='green', zsort='average', alpha=0.5, edgecolor='none', label=algo + ' long') 
                #lines3.append(line)
                ydepth=np.empty(len(self.bars.index))
                ydepth.fill(count)
                zpos=np.empty(len(self.bars.index))
                zpos.fill(0)
                colordepth=np.empty(len(self.bars.index)).astype(str)
                colordepth.fill('green')
                colordepth[np.array(self.signals[algo]) < 0]='red'
                line=ax3.bar3d(ydepth, mpl.dates.date2num(self.bars.index.to_pydatetime()), 
                     zpos, 
                     0.5,0.5,abs(np.array(self.signals[algo])),
                   color=colordepth, zsort='average', alpha=0.5, edgecolor='none')
                lines3.append(line)
                #, zdir='y', alpha=0.5, edgecolor='none', label=algo + ' short') 
                #ydepth=np.empty(len(mpl.dates.date2num(self.bars.index.to_pydatetime())[np.array(self.signals[algo]) < 0]))
                #ydepth.fill(count)                
                #line=ax3.bar3d(mpl.dates.date2num(self.bars.index.to_pydatetime())[np.array(self.signals[algo]) < 0], 
                #    ydepth, np.array(self.signals[algo])[np.array(self.signals[algo]) < 0], 
                #    1,1,1,
                #    color='red', zsort='average',alpha=0.5, edgecolor='none', label=algo + ' short') 
                #lines3.append(line)
                count = count - 1
            #ax3.yaxis.set_ticks(np.arange(mpl.dates.date2num(self.bars.index.to_pydatetime())[0],mpl.dates.date2num(self.bars.index.to_pydatetime())[-1],
            #                              (mpl.dates.date2num(self.bars.index.to_pydatetime())[-1]-
            #                              mpl.dates.date2num(self.bars.index.to_pydatetime())[-2])*3))
            
            ax3.set_xlabel('Signals', fontsize=8)
            green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
            red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
            ax3.legend([green_proxy,red_proxy],['long','short'], loc='upper left', fontsize=8) 
        
        lines4=list()
        def draw_pl():
            for line in lines4:
                    line.remove()
            del lines4[:]
            count = len(self.ranking)
            for [algo,equity] in self.ranking:
                if not color2.has_key(algo):
                    color2[algo]='#'+'%06X' % randint(0, 0xFFFFFF)
                    
                ydepth=np.empty(len(self.bars.index))
                ydepth.fill(count)
                zpos=np.empty(len(self.bars.index))
                zpos.fill(0)
                
                colordepth=np.empty(len(self.bars.index)).astype(str)
                colordepth.fill('green')
                colordepth[np.array(self.returns[algo]['profit']) < 0]='red'
                
                line=ax4.bar3d(ydepth, mpl.dates.date2num(self.bars.index.to_pydatetime()),
                     zpos, 
                       0.5,0.5,abs(np.array(self.returns[algo]['profit'])),
                   color=colordepth, zsort='average', alpha=0.5, edgecolor='none')
                lines4.append(line)
                
                #line, =ax[1].plot(self.returns[algo]['total'], color='b', lw=3.) 
                
                count = count - 1
            ax4.set_zlabel('PL in $', fontsize=8)
            green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
            red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
            ax4.legend([green_proxy,red_proxy],['profit','loss'], loc='upper left', fontsize=8) 
            #ax4.set_ylabel('Date', fontsize=8)
            
        def draw_arrow():
            r = 20  # or whatever fits you
            count = len(self.ranking)
            for [algo,equity] in self.ranking:
                x1, y1, _ = proj3d.proj_transform(count, mpl.dates.date2num(self.bars.index.to_pydatetime())[-1],
                                                  self.bars['Close'][-1], ax1.get_proj())
                
                if self.returns[algo]['profit'][-1] > 0:
                    
                   line= ax1.annotate('Chg: ' + str(round(self.returns[algo]['price_diff'][-1],2)) + '\nPL: $' + 
                        str(round(self.returns[algo]['profit'][-1])), 
                        xycoords='data', xy=(x1,y1),
                        xytext=(0, -55), textcoords='offset points',ha='center',fontsize=8,
                        arrowprops=dict(facecolor=color2[algo], shrink=0.1), backgroundcolor='white'
                        )
                   lines1.append(line)
                       
                else:
                    line=ax1.annotate('Chg: ' + str(round(self.returns[algo]['price_diff'][-1],2)) + '\nPL: $' + 
                        str(round(self.returns[algo]['profit'][-1],2)), 
                        xycoords='data', xy=(x1,y1),
                        xytext=(0, 55), textcoords='offset points',ha='center',fontsize=8,
                        arrowprops=dict(facecolor=color2[algo], shrink=0.1),backgroundcolor='white'
                        )
                    lines1.append(line)
                
                if self.nextSig[algo] > 0:
                    x3, y3, _ = proj3d.proj_transform(count, mpl.dates.date2num(self.bars.index.to_pydatetime())[-1],
                                                   0, ax3.get_proj())
                
                    line=ax3.annotate(algo, 
                        xycoords='data', xy=(x3,y3),
                        xytext=(0, -30), textcoords='offset points', ha='center',fontsize=8, 
                        arrowprops=dict(fc='green',ec='green', shrinkA=0.1,shrinkB=0.1,arrowstyle="-|>, head_length=1, head_width=0.5"), backgroundcolor='white'
                        )
                    lines3.append(line)
                else:
                    x3, y3, _ = proj3d.proj_transform(count, mpl.dates.date2num(self.bars.index.to_pydatetime())[-1],
                                                   0, ax3.get_proj())
                
                    line=ax3.annotate(algo, 
                        xycoords='data', xy=(x3,y3),
                        xytext=(0, -30), textcoords='offset points', ha='center',fontsize=8, 
                        arrowprops=dict(fc='red',ec='red', shrinkA=0.1,shrinkB=0.1,arrowstyle="<|-, head_length=1, head_width=0.5"), backgroundcolor='white'
                        )
                    lines3.append(line)
               
                x3_2, y3_2, _ = proj3d.proj_transform(count, 
                                            mpl.dates.date2num(self.bars.index.to_pydatetime())[-1],
                                                  0, ax2.get_proj())
                        
                if (self.returns[algo]['price_diff'][-1] > 0 and self.lastSig[algo] > 0) or \
                   (self.returns[algo]['price_diff'][-1] < 0 and self.lastSig[algo] < 0):
                    coord=[-16,-25]
                    if self.lastSig[algo] > 0:
                        coord=[-16,-25]
                    else:
                        coord=[-16,-25]
                    line=ax2.annotate('O',
                        xycoords='data', xy=(x3_2,y3_2),
                        xytext=coord, ha='center',
                            textcoords='offset points', fontsize=20, color='green')
                            #arrowprops=dict(facecolor='green', shrink=0.001))
                    lines1.append(line)
                elif self.lastSig[algo] != 0:
                    coord=[-16,-25]
                    if self.lastSig[algo] > 0:
                        coord=[-16,-25]
                    else:
                        coord=[-16,-25]
                    #xy=(self.bars.index[-2],self.returns['total'][-2])
                    line=ax2.annotate('X',
                        xycoords='data', xy=(x3_2,y3_2),
                        xytext=coord, ha='center',
                            textcoords='offset points', fontsize=20, color='red')
                            #arrowprops=dict(facecolor='red', shrink=0.001))
                    lines1.append(line)
                count = count - 1
            
                    
        
        def update(*args):
            print "Updating Graph"
            #lines[0].set_ydata(self.bars['Close'])
            #lines[1].set_ydata(self.returns['total'])
            draw_close()
            draw_equity()
            draw_signals()
            draw_pl()
            draw_arrow()
            #line, =ax[0].plot(self.bars['Close'], color='r', lw=3.)   
            #line, =ax[1].plot(self.returns['total'], color='b', lw=3.) 
            #line=ax[2].bar(np.array(self.bars.index)[np.array(self.signals['signal']) > 0], 
            #    np.array(self.signals['signal'])[np.array(self.signals['signal']) > 0], width, color='g', edgecolor='none') 
            #line=ax[2].bar(np.array(self.bars.index)[np.array(self.signals['signal']) < 0], 
            #    np.array(self.signals['signal'])[np.array(self.signals['signal']) < 0], width, color='r', edgecolor='none') 
            #ax1.set_title('(' + str(self.bars.index[0]) + '-' + str(self.bars.index[-1]) +')')
            #for angle in range(0,360):
            #    ax3.view_init(30, angle)
            #    plt.draw()
            
            fig.fmt_ydata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            #fig.autofmt_ydate()
        
            #plt.draw()
            # Return modified artists
            lines=list()
            lines.extend(lines1)
            lines.extend(lines2)
            lines.extend(lines3)
            lines.extend(lines4)
            ax2.set_title('Trading Period [' + str(self.bars.index[0]) + ' - ' + str(self.bars.index[-1]) +']', 
                                       fontsize=12,y=1.05)
            
            finaldate=mpl.dates.date2num(self.bars.index.to_pydatetime())[-1]+1
            begindate=mpl.dates.date2num(self.bars.index.to_pydatetime())[0]
            freq=(finaldate - begindate)/3
            ax1.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
            ax2.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
            ax3.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
            ax4.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
            print "Finished Updating Graph"
            return lines
            
        #fig.show()
        #plt.ion()
        plt.gca().invert_yaxis()
        
        draw_close()
        draw_equity()
        draw_signals()
        draw_pl()
        draw_arrow()  
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        ax4.tick_params(axis='both', which='major', labelsize=8)
        #loc = ax3.w_yaxis.get_major_locator()
        #loc.maxticks[DAILY] = 24
        ax4.yaxis_date()
        ax4.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        #for label in ax4.get_yticklabels()[::2]:
        #    label.set_visible(False)
        ax3.yaxis_date()
        ax3.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax3.xaxis.label.set_fontsize(8)
        #for label in ax3.get_yticklabels()[::2]:
        #    label.set_visible(False)
        #ax3.set_yscale('log')
        ax2.yaxis_date()
        ax2.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax2.xaxis.label.set_fontsize(8)
        #for label in ax2.get_yticklabels()[::2]:
        #    label.set_visible(False)
        #ax2.set_yscale('log')
        ax1.yaxis_date()
        ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax1.xaxis.label.set_fontsize(8)
        
        finaldate=mpl.dates.date2num(self.bars.index.to_pydatetime())[-1]+1
        begindate=mpl.dates.date2num(self.bars.index.to_pydatetime())[0]
        freq=(finaldate - begindate)/3
        ax1.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
        ax2.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
        ax3.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
        ax4.yaxis.set_ticks(np.arange(begindate, finaldate, freq))
        #for label in ax1.get_yticklabels()[::2]:
        #    label.set_visible(False)
        
        #ax1.set_title()
        #ax1.set_yscale('log')
            #ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
           
        #fig.autofmt_xdate()
        #fig.fmt_ydata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        #fig.autofmt_ydate()
        
        ax2.zaxis.set_major_formatter(FormatStrFormatter('$%.0f'))
        ax4.zaxis.set_major_formatter(FormatStrFormatter('$%.0f'))
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        ax3.invert_xaxis()
        ax4.invert_xaxis()
        #plt.gca().invert_xaxis()
        anim = animation.FuncAnimation(fig, update, interval=5)
        #fig.show()
        #plt.draw()
        plt.show()
        
        #while 1:
        #    update()
        #    plt.draw()
        #    time.sleep(5)
        

