import datetime
import pandas as pd

high={}
low={}
close={}
open={}
date={}
df=pd.read_csv('./test.csv')
dataSet=df.sort_values(by='Date')
dataSet.to_csv('./test2.csv')

