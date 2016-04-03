import re
import os

def get_btc_list():
    dataPath='./data/from_IB/'
    files = [ f for f in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath,f)) ]
    btcList=list()
    for file in files:
            if re.search(r'BTCUSD', file):
                (inst, ext)=file.split('.')
                btcList.append(inst)
                print 'Found ' + inst
    return btcList
get_btc_list()
