import datetime
import pandas as pd

high={}
low={}
close={}
open={}
date={}
df=pd.read_csv('./bitcoin_coindesk.csv')
for i in df.index:
    row=df.ix[i]
    hour=datetime.datetime.fromtimestamp(
        int(row[0])
    ).strftime('%Y-%m-%d %H')    
    if open.has_key(hour):
	if row[1] > high[hour]:
		high[hour]=row[1]
	if row[1] < low[hour]:
		low[hour]=row[1]
	close[hour]=row[1]
    else:
	open[hour]=row[1]
	high[hour]=row[1]
	low[hour]=row[1]
	close[hour]=row[1]	
	date[hour]=str(hour) + ":00"
dataSet=pd.DataFrame({'Date':date.values(), 'Open':open.values(), 'High':high.values(),
		      'Low':low.values(), 'Close':close.values()}, columns=['Date','Open','High','Low','Close'])	
dataSet.to_csv('./test.csv')

