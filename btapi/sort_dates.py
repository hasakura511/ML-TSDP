import datetime
import pandas as pd

def sort_dates(infile, outfile):
    high={}
    low={}
    close={}
    open={}
    date={}
    df=pd.read_csv(infile, index_col=['Date'])
    dataSet=df.sort_values(by='Date')
    dataSet.to_csv(outfile)

