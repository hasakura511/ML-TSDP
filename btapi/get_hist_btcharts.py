import requests
import urllib
import urllib2

def get_hist_btcharts(symbol):
    #url='http://api.bitcoincharts.com/v1/csv/'
    #http://www.quandl.com/markets/bitcoin
    url = 'http://api.bitcoincharts.com/v1/trades.csv'
    values = {'symbol' : symbol} #, 'start' : '1420121275'}
   
    response = requests.get(url, params=values, json=values);
    #print response.text;
    return response.text;
