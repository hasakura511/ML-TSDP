### Quantiacs Mean Reversion Trading System Example
# import necessary Packages below:
import numpy 

##### Do not change this function definition #####
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    #print DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings
    nMarkets=CLOSE.shape[1]
    pos=numpy.ones((1,nMarkets))
    pos=pos/numpy.sum(abs(pos))
    #print pos
    return pos, settings


##### Do not change this function definition #####
def mySettings():
    # Default competition and evaluation mySettings
    settings= {}

    # S&P 100 stocks
    '''
    settings['markets']=\
    ['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL', \
    'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C', \
    'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',\
    'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE', \
    'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM', \
    'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON', \
    'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM', \
    'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP', \
    'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM', \
    ]
    '''
    settings['markets']=['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT',\
        'F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY',\
        'F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O',\
        'F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM',\
        'F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ']

    settings['lookback']= 500
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings

