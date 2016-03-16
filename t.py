from os import listdir
from os.path import isfile, join
import re

dataPath='./data/paper/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('stratBTC')
tradesearch=re.compile('trade')
c2search=re.compile('c2')
for file in files:
	if re.search(btcsearch, file):
		if re.search(tradesearch, file):
			if re.search(c2search, file):
				print "C2" + file;
			else:
				print "ib " + file
