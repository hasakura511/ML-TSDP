import datetime
import pandas as pd
import numpy as np

def raw_to_ohlc_from_csv(infile, outfile):
    df=pd.read_csv(infile)
    return raw_to_ohlc(df, outfile)
    

def raw_to_ohlc(df, outfile):
    high={}
    low={}
    close={}
    open={}
    date={}
    volume={}

    for i in df.index:
        row=df.ix[i]
        if len(str(row[0])) > 0:
            hour=datetime.datetime.fromtimestamp(
                int(row[0])
            ).strftime('%Y-%m-%d %H') 
            
            if open.has_key(hour):
                
            	if row[1] > high[hour]:
            		high[hour]=row[1]
            	if row[1] < low[hour]:
            		low[hour]=row[1]
            	close[hour]=row[1]
            	volume[hour]=volume[hour] + row[2]
            else:
            	open[hour]=row[1]
            	high[hour]=row[1]
            	low[hour]=row[1]
            	close[hour]=row[1]	
            	date[hour]=str(hour) + ":00"
            	volume[hour]=row[2]
    dataSet=pd.DataFrame({'Date':date.values(), 'Open':open.values(), 'High':high.values(),
    		      'Low':low.values(), 'Close':close.values(), 'Volume':volume.values()}, columns=['Date','Open','High','Low','Close','Volume'])	
    dataSet=dataSet.sort_values(by='Date')
    dataSet=dataSet.set_index('Date')
    dataSet.to_csv(outfile)
    return dataSet

def raw_to_ohlc_min_from_csv(infile, outfile):
    df=pd.read_csv(infile)
    raw_to_ohlc_min(df)

def raw_to_ohlc_min(df, outfile):     
    high={}
    low={}
    close={}
    open={}
    date={}
    volume={}

    for i in df.index:
        row=df.ix[i]
        if np.any(np.isnan(row)) or np.all(np.isfinite(row)):
                pass
        hour=datetime.datetime.fromtimestamp(
            int(row[0])
        ).strftime('%Y-%m-%d %H:%M')    
        if len(str(hour)) > 0:
                    
            if open.has_key(hour):
                
            	if row[1] > high[hour]:
            		high[hour]=row[1]
            	if row[1] < low[hour]:
            		low[hour]=row[1]
            	close[hour]=row[1]
            	volume[hour]=volume[hour] + row[2]
            else:
            	open[hour]=row[1]
            	high[hour]=row[1]
            	low[hour]=row[1]
            	close[hour]=row[1]	
            	date[hour]=str(hour)
            	volume[hour]=row[2]
    dataSet=pd.DataFrame({'Date':date.values(), 'Open':open.values(), 'High':high.values(),
    		      'Low':low.values(), 'Close':close.values(), 'Volume':volume.values()}, columns=['Date','Open','High','Low','Close','Volume'])	
    
    dataSet=dataSet.sort_values(by='Date')
    dataSet=dataSet.set_index(['Date'])
    dataSet.to_csv(outfile)
    return dataSet
