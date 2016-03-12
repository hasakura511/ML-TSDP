
import numpy as np
import pandas as pd
from ibapi.get_feed import get_feed, get_realtimebar
from c2api.place_order import place_order as place_c2order

def get_ibfeed():
	get_feed('EUR','USD','IDEALPRO','CASH')

def get_ibrtbar():
     get_realtimebar('EUR','USD','IDEALPRO','CASH')

get_ibrtbar()
