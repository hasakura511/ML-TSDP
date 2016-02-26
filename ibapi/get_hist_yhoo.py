from yahoo_finance import Share
from pprint import pprint

yahoo = Share('YHOO')
pprint(yahoo.get_historical('2014-04-25', '2014-04-29'))
