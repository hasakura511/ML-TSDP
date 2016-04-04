import seitoolz.bars as bars

def onBar(bar, symbols):
	print "==============="
	for symbol in symbols:
		print "OnBar: " + symbol + '\n'
	print "==============="

bars.get_last_bar(['EURUSD','USDJPY'], 'Close', onBar)

