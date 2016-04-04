import seitoolz.bars as bars

def onBar(bar, symbols):
	print "==============="
	print "Date: " + bar['Date'] + '\n'
	for symbol in symbols:
		print "OnBar: " + symbol + '\n'
		print "OnBar: " + bar[symbol] + '\n'
	print "==============="

bars.get_last_bar(['EURUSD','USDJPY'], 'Close', onBar)

