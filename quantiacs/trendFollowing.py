'''
### Quantiacs Trend Following Trading System Example

# import necessary Packages below:
import numpy

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    #This system uses trend following techniques to allocate capital into the desired equities

    nMarkets=CLOSE.shape[1]

    periodLong=200
    periodShort=40

    smaLong=numpy.nansum(CLOSE[-periodLong:,:],axis=0)/periodLong
    smaRecent=numpy.nansum(CLOSE[-periodShort:,:],axis=0)/periodShort

    longEquity= numpy.array(smaRecent > smaLong)
    shortEquity= ~longEquity

    pos=numpy.zeros((1,nMarkets))
    pos[0,longEquity]=1
    pos[0,shortEquity]=-1

    weights = pos/numpy.nansum(abs(pos))

    return weights, settings


def mySettings():
    #Define your trading system settings here
    settings= {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL', \
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C', \
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',\
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE', \
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM', \
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON', \
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM', \
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP', \
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets']  = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CD',  \
    'F_CL', 'F_DJ', 'F_EC', 'F_ES', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_LC', \
    'F_LN', 'F_NG', 'F_NQ', 'F_RB', 'F_S', 'F_SF', 'F_SI', 'F_SM', 'F_SP', \
    'F_TY', 'F_US', 'F_W', 'F_YM']


    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings
'''

### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import numpy

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    nMarkets=CLOSE.shape[1]

    periodLong=200
    periodShort=40

    smaLong=numpy.nansum(CLOSE[-periodLong:,:],axis=0)/periodLong
    smaRecent=numpy.nansum(CLOSE[-periodShort:,:],axis=0)/periodShort

    longEquity= numpy.array(smaRecent > smaLong)
    shortEquity= ~longEquity

    pos=numpy.zeros((1,nMarkets))
    pos[0,longEquity]=1
    pos[0,shortEquity]=-1

    weights = pos/numpy.nansum(abs(pos))

    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}
    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL', \
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C', \
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',\
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE', \
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM', \
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON', \
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM', \
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP', \
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets']  = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CD',  \
    'F_CL', 'F_DJ', 'F_EC', 'F_ES', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_LC', \
    'F_LN', 'F_NG', 'F_NQ', 'F_RB', 'F_S', 'F_SF', 'F_SI', 'F_SM', 'F_SP', \
    'F_TY', 'F_US', 'F_W', 'F_YM']

    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings
