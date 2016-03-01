from datetime import datetime
from threading import Event

from swigibpy import EWrapper, EPosixClientSocket, Contract


WAIT_TIME = 30.0
p_open=[];
p_high=[];
p_low=[];
p_close=[];
p_volume=[];
p_chg=[];

###


class HistoricalDataExample(EWrapper):
    '''Callback object passed to TWS, these functions will be called directly
    by TWS.
    '''

    def __init__(self):
        super(HistoricalDataExample, self).__init__()
        self.got_history = Event()

    def orderStatus(self, id, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFilledPrice, clientId, whyHeld):
        pass

    def openOrder(self, orderID, contract, order, orderState):
        pass

    def nextValidId(self, orderId):
        '''Always called by TWS but not relevant for our example'''
        pass

    def openOrderEnd(self):
        '''Always called by TWS but not relevant for our example'''
        pass

    def managedAccounts(self, openOrderEnd):
        '''Called by TWS but not relevant for our example'''
        pass

    def historicalData(self, reqId, date, open, high,
                       low, close, volume,
                       barCount, WAP, hasGaps):

        if date[:8] == 'finished':
            print("History request complete")
            self.got_history.set()
        else:
	    chg=0;
	    chgpt=0;
	    if len(p_close) > 0:
	    	chgpt=close-p_close[-1];
		chg=chgpt/p_close[-1];
		
            p_open.append(open);
            p_high.append(high);
            p_low.append(low);
            p_close.append(close);
            p_volume.append(volume);
	    date = datetime.strptime(date, "%Y%m%d").strftime("%d %b %Y")
            print(("History %s - Open: %s, High: %s, Low: %s, Close: "
                   "%s, Volume: %d, Change: %s, Net: %s") % (date, open, high, low, close, volume, chgpt, chg))

# Instantiate our callback object
callback = HistoricalDataExample()

# Instantiate a socket object, allowing us to call TWS directly. Pass our
# callback object so TWS can respond.

tws = EPosixClientSocket(callback)
#tws = EPosixClientSocket(callback, reconnect_auto=True)
# Connect to tws running on localhost
if not tws.eConnect("", 7496, 111):
    raise RuntimeError('Failed to connect to TWS')

# Simple contract for GOOG
contract = Contract()
contract.exchange = "IDEALPRO"
contract.symbol = "USD"
contract.secType = "CASH"
contract.currency = "JPY"
today = datetime.today()

print("Requesting historical data for %s" % contract.symbol)

# Request some historical data.
tws.reqHistoricalData(
    1,                                         # tickerId,
    contract,                                   # contract,
    today.strftime("%Y%m%d %H:%M:%S %Z"),       # endDateTime,
    "1 W",                                      # durationStr,
    "1 day",                                    # barSizeSetting,
    "ASK",                                   # whatToShow,
    1,                                          # useRTH,
    1                                          # formatDate
)

print("\n====================================================================")
print(" History requested, waiting %ds for TWS responses" % WAIT_TIME)
print("====================================================================\n")


try:
    callback.got_history.wait(timeout=WAIT_TIME)
except KeyboardInterrupt:
    pass
finally:
    if not callback.got_history.is_set():
        print('Failed to get history within %d seconds' % WAIT_TIME)

    print("\nDisconnecting...")
    tws.eDisconnect()
