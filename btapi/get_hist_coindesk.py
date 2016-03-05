import requests
import urllib
import urllib2

def get_hist_coindesk():
    url = 'https://blockchain.info/charts/market-price?timespan=60days&format=json'
    values = {'timespan' : '60days',
              'format' : 'json'}
    
    response = requests.get(url, params=values, json=values);
    return response.text;
