import sqlite3
import time
import pandas as pd
from pandas.io import sql

start_time = time.time()

#table for account master

#readConn = sqlite3.readConnect('futures_master.sqlite')
dbPath='D:/ML-TSDP/web/tsdp/db.sqlite3'
writeConn = sqlite3.connect(dbPath)
dbPath2='D:/ML-TSDP/data/futures.sqlite3'
readConn = sqlite3.connect(dbPath2)
cur = readConn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
    

for table_name in tables:
    table_name = table_name[0]
    table = pd.read_sql_query("SELECT * from %s" % table_name, readConn)
    #table.to_sql(table_name + '.csv', index_label='index')
    table.to_sql(name=table_name, if_exists='replace', con=writeConn, index=False)
    
fDict = pd.read_csv('C:/Users/Hidemi/Desktop/Python/TSDP/ml/futuresdict.csv')
fDict.index.name='id'
fDict.to_sql(name='Dictionary', if_exists='replace', con=writeConn, index=True, index_label='id')
'''
fDict.CSIsym.to_csv('data\keys\dictionary.csv',index=True, header=True)
#cur.execute('INSERT OR IGNORE INTO Dictionary (CSIsym)  VALUES ( ? )', ( 'TEST2', ) )
#readConn.commit()

currenciesDF = pd.read_csv('D:/ML-TSDP/data/currenciesATR.csv', index_col=0)
date = currenciesDF.index.name.split()[0].replace('-','')
#add date
currenciesDF['Date']=date
currenciesDF.index.name='pairs'
currenciesDF= currenciesDF.reset_index()
currenciesDF.to_sql(name='currenciesATR', if_exists='replace', con=readConn, index=True, index_label='id')

currenciesDF = currenciesDF.reset_index()
currenciesDF.index.name='id'
currenciesDF.pairs.to_csv('data\keys\currenciesATR.csv',index=True, header=True)
    



futuresDF = pd.read_csv('D:/ML-TSDP/data/futuresATR.csv', index_col=0)
date = futuresDF.index.name.split()[0].replace('-','')
#fkey_dict=pd.read_csv('data\keys\dictionary.csv', index_col=0)
#fkey_curr=pd.read_csv('data\keys\currenciesATR.csv', index_col=0)
#add date
futuresDF['Date']=date
#add currency

#add foreign key
#for sym in futuresDF.index:
#    futuresDF.set_value(sym, 'dictionary_id', fkey_dict[fkey_dict.CSIsym==sym].index[0])
#futuresDF.dictionary_id=futuresDF.dictionary_id.astype(int)
futuresDF.index.name='CSIsym'
futuresDF = futuresDF.reset_index()
futuresDF.to_sql(name='futuresATR', if_exists='replace', con=readConn, index=True, index_label='id')
#need to manually update keys
#futuresDF.to_sql(name='futuresATR', if_exists='append', con=readConn, index=True, index_label='id')
#futuresDF.CSIsym.to_csv('data\keys\ATRdict.csv',index=True)

'''
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes '