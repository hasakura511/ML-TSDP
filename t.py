import seitoolz.bars as bars

def onBar(bar, symbols):
	print "==============="
	for symbol in symbols:
		print "OnBar: " + symbol + '\n'
	print "==============="

get_last_bars(['EURUSD','USDJPY'], 'Close', onBar)

