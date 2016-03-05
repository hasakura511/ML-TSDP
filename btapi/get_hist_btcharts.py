import requests
import urllib
import urllib2

def get_hist_blockchain():
    #url='http://api.bitcoincharts.com/v1/csv/'
    #http://www.quandl.com/markets/bitcoin
    url = 'https://blockchain.info/charts/market-price?timespan=60days&format=json'
    values = {'timespan' : '60days',
              'format' : 'json'}
    response = requests.get(url, params=values, json=values);
    print response.text;
    return response.text;
