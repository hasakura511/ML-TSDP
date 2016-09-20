import seitoolz.bars as bars

def onBar(bar, symbols):
	bar=bar.iloc[-1]
	print "==============="
	print "Date: " + bar['Date'] + '\n'
	for symbol in symbols:
		print "OnBar: " + symbol + '\n'
		print "OnBar: " + str(bar[symbol]) + '\n'
	print "==============="

bars.get_last_bars(['EURUSD','USDJPY'], 'Close', onBar)
#bars.get_last_bars(['BTCUSD_bitfinexUSD','BTCUSD_bitstampUSD'], 'Close', onBar)
