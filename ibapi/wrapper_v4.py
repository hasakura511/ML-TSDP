from swigibpy import EWrapper
import time
import datetime
import numpy as np
import pandas as pd
import random
from swigibpy import EPosixClientSocket, ExecutionFilter, CommissionReport, Execution, Contract
from swigibpy import Order as IBOrder
from IButils import bs_resolve, action_ib_fill
from pytz import timezone
from threading import Event

MAX_WAIT_SECONDS=10
MEANINGLESS_NUMBER=1729

## This is the reqId IB API sends when a fill is received
FILL_CODE=-1
rtbar={}
rtdict={}
rthist={}
rtfile={}

def return_IB_connection_info():
    """
    Returns the tuple host, port, clientID required by eConnect
   
    """
   
    host=""
   
    port=7496
    clientid=random.randint(100,9999)
    
    return (host, port, clientid)

class IBWrapper(EWrapper):
    """
    Callback object passed to TWS, these functions will be called directly by the TWS or Gateway.
    """
    global rtbar
    global rtdict
    
    def init_error(self):
        setattr(self, "flag_iserror", False)
        setattr(self, "error_msg", "")

    def error(self, id, errorCode, errorString):
        """
        error handling, simple for now
       
        Here are some typical IB errors
        INFO: 2107, 2106
        WARNING 326 - can't connect as already connected
        CRITICAL: 502, 504 can't connect to TWS.
            200 no security definition found
            162 no trades
        """
        ## Any errors not on this list we just treat as information
        ERRORS_TO_TRIGGER=[201, 103, 502, 504, 509, 200, 162, 420, 2105, 1100, 478, 201, 399]
       
        if errorCode in ERRORS_TO_TRIGGER:
            errormsg="IB error id %d errorcode %d string %s" %(id, errorCode, errorString)
            print errormsg
            setattr(self, "flag_iserror", True)
            setattr(self, "error_msg", True)
           
        ## Wrapper functions don't have to return anything
       
    """
    Following methods will be called, but we don't use them
    """
       
    def managedAccounts(self, openOrderEnd):
        pass

    def orderStatus(self, reqid, status, filled, remaining, avgFillPrice, permId,
            parentId, lastFilledPrice, clientId, whyHeld):
        pass

    def commissionReport(self, commission):
        print 'Commission %s %s P&L: %s' % (commission.currency,
                                            commission.commission,
                                            commission.realizedPNL)
        filldata=self.data_fill_data
        
        #if reqId not in filldata.keys():
        #    filldata[reqId]={}
            
        execid=commission.execId
        
        execdetails=filldata[execid]
        execdetails['commission']=commission.commission
        execdetails['commission_currency']=commission.currency
        execdetails['realized_PnL']=commission.realizedPNL
        execdetails['yield_redemption_date']=commission.yieldRedemptionDate
        filldata[execid]=execdetails
        self.data_fill_data=filldata
        

    """
    get stuff
    """

    def init_fill_data(self):
        setattr(self, "data_fill_data", {})
        setattr(self, "flag_fill_data_finished", False)

    def add_fill_data(self, reqId, execdetails):
        #if "data_fill_data" not in dir(self):
        #    filldata=execdetails
        #else:
        filldata=self.data_fill_data

        #if reqId not in filldata.keys():
        #    filldata[reqId]={}
            
        #execid=execdetails['execid']
        
        filldata[execdetails['execid']]=execdetails
                        
        setattr(self, "data_fill_data", filldata)


    def execDetails(self, reqId, contract, execution):
        
        """
        This is called if 
        
        a) we have submitted an order and a fill has come back
        b) We have asked for recent fills to be given to us 
        
        We populate the filldata object and also call action_ib_fill in case we need to do something with the 
          fill data 
        
        See API docs, C++, SocketClient Properties, Contract and Execution for more details 
        """
        reqId=int(reqId)
       
        execid=execution.execId
        exectime=execution.time
        thisorderid=int(execution.orderId)
        account=execution.acctNumber
        exchange=execution.exchange
        permid=execution.permId
        avgprice=execution.price
        cumQty=execution.cumQty
        clientid=execution.clientId
        symbol=contract.symbol
        expiry=contract.expiry
        side=execution.side
        #commission=execution.commission
        currency=contract.currency
        
        
        execdetails=dict( symbol_currency=str(currency), side=str(side), times=str(exectime), orderid=str(thisorderid), qty=int(cumQty), price=float(avgprice), symbol=str(symbol), expiry=str(expiry), clientid=str(clientid), execid=str(execid), account=str(account), exchange=str(exchange), permid=int(permid))
        
        if reqId==FILL_CODE:
            ## This is a fill from a trade we've just done
            action_ib_fill(execdetails)
            
        else:
            ## This is just execution data we've asked for
            self.add_fill_data(reqId, execdetails)

        
            
    def execDetailsEnd(self, reqId):
        """
        No more orders to look at if execution details requested
        """

        setattr(self, "flag_fill_data_finished", True)
            

    def init_openorders(self):
        setattr(self, "data_order_structure", {})
        setattr(self, "flag_order_structure_finished", False)

    def add_order_data(self, orderdetails):
        if "data_order_structure" not in dir(self):
            orderdata={}
        else:
            orderdata=self.data_order_structure

        orderid=orderdetails['orderid']
        orderdata[orderid]=orderdetails
                        
        setattr(self, "data_order_structure", orderdata)


    def openOrder(self, orderID, contract, order, orderState):
        """
        Tells us about any orders we are working now
        
        Note these objects are not persistent or interesting so we have to extract what we want
        
        
        """
        
        ## Get a selection of interesting things about the order
        orderdetails=dict(symbol=contract.symbol , expiry=contract.expiry,  qty=int(order.totalQuantity) , 
                       side=order.action , orderid=int(orderID), clientid=order.clientId ) 
        
        self.add_order_data(orderdetails)

    def openOrderEnd(self):
        """
        Finished getting open orders
        """
        setattr(self, "flag_order_structure_finished", True)


    def init_nextvalidid(self):
        setattr(self, "data_brokerorderid", None)


    def nextValidId(self, orderId):
        """
        Give the next valid order id 
        
        Note this doesn't 'burn' the ID; if you call again without executing the next ID will be the same
        """
        
        self.data_brokerorderid=orderId


    def init_contractdetails(self, reqId):
        if "data_contractdetails" not in dir(self):
            dict_contractdetails=dict()
        else:
            dict_contractdetails=self.data_contractdetails
        
        dict_contractdetails[reqId]={}
        setattr(self, "flag_finished_contractdetails", False)
        setattr(self, "data_contractdetails", dict_contractdetails)
        

    def contractDetails(self, reqId, contractDetails):
        """
        Return contract details
        
        If you submit more than one request watch out to match up with reqId
        """
        
        contract_details=self.data_contractdetails[reqId]

        contract_details["contractMonth"]=contractDetails.contractMonth
        contract_details["liquidHours"]=contractDetails.liquidHours
        contract_details["longName"]=contractDetails.longName
        contract_details["minTick"]=contractDetails.minTick
        contract_details["tradingHours"]=contractDetails.tradingHours
        contract_details["timeZoneId"]=contractDetails.timeZoneId
        contract_details["underConId"]=contractDetails.underConId
        contract_details["evRule"]=contractDetails.evRule
        contract_details["evMultiplier"]=contractDetails.evMultiplier

        contract2 = contractDetails.summary

        contract_details["expiry"]=contract2.expiry

        contract_details["exchange"]=contract2.exchange
        contract_details["symbol"]=contract2.symbol
        contract_details["secType"]=contract2.secType
        contract_details["currency"]=contract2.currency

    def contractDetailsEnd(self, reqId):
        """
        Finished getting contract details
        """
        
        setattr(self, "flag_finished_contractdetails", True)



    ## portfolio

    def init_portfolio_data(self):
        if "data_portfoliodata" not in dir(self):
            setattr(self, "data_portfoliodata", [])
        if "data_accountvalue" not in dir(self):
            setattr(self, "data_accountvalue", [])
            
        
        setattr(self, "flag_finished_portfolio", False)
        

    def updatePortfolio(self, contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName):
        """
        Add a row to the portfolio structure
        """

        portfolio_structure=self.data_portfoliodata
                
        portfolio_structure.append((contract.symbol, contract.expiry, position, marketPrice, marketValue, averageCost, 
                                    unrealizedPNL, realizedPNL, accountName, contract.currency))

    ## account value
    
    def updateAccountValue(self, key, value, currency, accountName):
        """
        Populates account value dictionary
        """
        account_value=self.data_accountvalue
        
        account_value.append((key, value, currency, accountName))
        

    def accountDownloadEnd(self, accountName):
        """
        Finished can look at portfolio_structure and account_value
        """
        setattr(self, "flag_finished_portfolio", True)

    
         
    def historicalData(self, reqId, date, open, high,
                       low, close, volume,
                       barCount, WAP, hasGaps):
        global rtbar
        global rtdict
        global rthist
        sym=rtdict[reqId]
        data=rtbar[reqId]
        if date[:8] == 'finished':
            print("History request complete")
            rthist[reqId].set()
            data=data.reset_index()
            data=data.sort_values(by='Date')  
            data=data.set_index('Date')
        else:
            data.loc[date] = [open,high,low,close,volume]
            print "History %s - Open: %s, High: %s, Low: %s, Close: %s, Volume: %d"\
                      % (date, open, high, low, close, volume)
        rtbar[reqId]=data
        

            #print(("History %s - Open: %s, High: %s, Low: %s, Close: "
            #       "%s, Volume: %d, Change: %s, Net: %s") % (date, open, high, low, close, volume, chgpt, chg));

        #return self.data

    def realtimeBar(self, reqId, time, open, high, low, close, volume, wap, count):

        """
        Note we don't use all the information here
        
        Just append close prices. 
        """
       
        global pricevalue
        global finished
        global rtbar
        global rtdict
        global rtfile
        sym=rtdict[reqId]
        data=rtbar[reqId]
        filename=rtfile[reqId]
        
        eastern=timezone('US/Eastern')
        
        time=datetime.datetime.fromtimestamp(
                    int(time), eastern
                ).strftime('%Y%m%d  %H:%M:00') 
        #time=time.astimezone(eastern).strftime('%Y-%m-%d %H:%M:00') 
        
        if time in data.index:
               
            quote=data.loc[time]
            if high > quote['High']:
                quote['High']=high
            if low < quote['Low']:
                quote['Low']=low
            quote['Close']=close
            quote['Volume']=quote['Volume'] + volume
            if quote['Volume'] < 0:
                quote['Volume'] = 0 
            data.loc[time]=quote
            #print "Update Bar: bar: sym: " + sym + " date:" + str(time) + "open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']) + ' wap:' + str(wap) + ' count:' + str(count)
        
        else:
            if len(data.index) > 1:
                data=data.reset_index()                
                data=data.sort_values(by='Date')  
                quote=data.iloc[-1]
                print "Close Bar: " + sym + " date:" + str(quote['Date']) + " open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']) + ' wap:' + str(wap) + ' count:' + str(count)
                data=data.set_index('Date')
                data.to_csv(filename)
            print "New Bar:   " + sym + " date:" + str(time) + " open: " + str(open) + " high:"  + str(high) + ' low:' + str(low) + ' close: ' + str(close) + ' volume:' + str(volume) + ' wap:' + str(wap) + ' count:' + str(count)
            
            data=data.reset_index().append(pd.DataFrame([[time, open, high, low, close, volume]], columns=['Date','Open','High','Low','Close','Volume'])).set_index('Date')
            
            
        rtbar[reqId]=data
        
        
        #pricevalue.append(close)

    def init_tickdata(self, TickerId):
        if "data_tickdata" not in dir(self):
            tickdict=dict()
        else:
            tickdict=self.data_tickdata

        tickdict[TickerId]=[np.nan]*4
        setattr(self, "data_tickdata", tickdict)


    def tickString(self, TickerId, field, value):
        marketdata=self.data_tickdata[TickerId]

        ## update string ticks

        tickType=field

        if int(tickType)==0:
            ## bid size
            marketdata[0]=int(value)
        elif int(tickType)==3:
            ## ask size
            marketdata[1]=int(value)

        elif int(tickType)==1:
            ## bid
            marketdata[0][2]=float(value)
        elif int(tickType)==2:
            ## ask
            marketdata[0][3]=float(value)
        print "ASK: " + str(marketdata[0]) + " ASKSIZE: " + str(marketdata[1]) +  "BID: " + str(marketdata[2]) + " BIDSIZE: " + str(marketdata[3])


    def tickGeneric(self, TickerId, tickType, value):
        marketdata=self.data_tickdata[TickerId]

        ## update generic ticks

        if int(tickType)==0:
            ## bid size
            marketdata[0]=int(value)
        elif int(tickType)==3:
            ## ask size
            marketdata[1]=int(value)

        elif int(tickType)==1:
            ## bid
            marketdata[2]=float(value)
        elif int(tickType)==2:
            ## ask
            marketdata[3]=float(value)
        print "ASK: " + str(marketdata[0]) + " ASKSIZE: " + str(marketdata[1]) +  "BID: " + str(marketdata[2]) + " BIDSIZE: " + str(marketdata[3])
        
        
           
    def tickSize(self, TickerId, tickType, size):
        
        ## update ticks of the form new size
        
        marketdata=self.data_tickdata[TickerId]

        
        if int(tickType)==0:
            ## bid
            marketdata[0]=int(size)
        elif int(tickType)==3:
            ## ask
            marketdata[1]=int(size)
        
        print "tickSize: ASKSIZE: " + str(marketdata[0]) +  " BIDSIZE: " + str(marketdata[1])

   
    def tickPrice(self, TickerId, tickType, price, canAutoExecute):
        ## update ticks of the form new price
        
        marketdata=self.data_tickdata[TickerId]
        
        if int(tickType)==1:
            ## bid
            marketdata[2]=float(price)
        elif int(tickType)==2:
            ## ask
            marketdata[3]=float(price)
        
        print "tickPrice: ASK: " + str(marketdata[2]) +  " BID: " + str(marketdata[3])

    def updateMktDepth(self, id, position, operation, side, price, size):
        """
        Only here for completeness - not required. Market depth is only available if you subscribe to L2 data.
        Since I don't I haven't managed to test this.
        
        Here is the client side call for interest
        
        tws.reqMktDepth(999, ibcontract, 9)
        
        """
        pass

        
    def tickSnapshotEnd(self, tickerId):
        
        print "No longer want to get %d" % tickerId

class IBclient(object):
    """
    Client object
    
    Used to interface with TWS for outside world, does all handling of streaming waiting etc
    
    Create like this
    callback = IBWrapper()
    client=IBclient(callback)
    We then use various methods to get prices etc
    """
    def __init__(self, callback):
        """
        Create like this
        callback = IBWrapper()
        client=IBclient(callback)
        """
        
        tws = EPosixClientSocket(callback)
        (host, port, clientid)=return_IB_connection_info()
        tws.eConnect(host, port, clientid)

        self.tws=tws
        self.cb=callback
        self.accountid=''

    
    def get_contract_details(self, ibcontract, reqId=MEANINGLESS_NUMBER):
    
        """
        Returns a dictionary of contract_details
        
        
        """
        
        self.cb.init_contractdetails(reqId)
        self.cb.init_error()
    
        self.tws.reqContractDetails(
            reqId,                                         # reqId,
            ibcontract,                                   # contract,
        )
    

        finished=False
        iserror=False
        
        start_time=time.time()
        while not finished and not iserror:
            finished=self.cb.flag_finished_contractdetails
            iserror=self.cb.flag_iserror
            
            if (time.time() - start_time) > MAX_WAIT_SECONDS:
                finished=True
                iserror=True
            pass
    
        contract_details=self.cb.data_contractdetails[reqId]
        if iserror or contract_details=={}:
            print self.cb.error_msg
            print "Problem getting details"
            return None
    
        return contract_details



    def get_next_brokerorderid(self):
        """
        Get the next brokerorderid
        """


        self.cb.init_error()
        self.cb.init_nextvalidid()
        

        start_time=time.time()
        
        ## Note for more than one ID change '1'
        self.tws.reqIds(1)

        finished=False
        iserror=False

        while not finished and not iserror:
            brokerorderid=self.cb.data_brokerorderid
            finished=brokerorderid is not None
            iserror=self.cb.flag_iserror
            if (time.time() - start_time) > MAX_WAIT_SECONDS:
                finished=True
            pass

        
        if brokerorderid is None or iserror:
            print self.cb.error_msg
            print "Problem getting next broker orderid"
            return None
        
        return brokerorderid


    def place_new_IB_order(self, ibcontract, trade, lmtPrice, orderType, orderid=None):
        """
        Places an order
        
        Returns brokerorderid
    
        raises exception if fails
        """
        iborder = IBOrder()
        iborder.action = bs_resolve(trade)
        iborder.lmtPrice = lmtPrice
        iborder.orderType = orderType
        iborder.totalQuantity = abs(trade)
        iborder.tif='DAY'
        iborder.transmit=True

        ## We can eithier supply our own ID or ask IB to give us the next valid one
        if orderid is None:
            print "Getting orderid from IB"
            orderid=self.get_next_brokerorderid()
            
        print "Using order id of %d" % orderid
    
         # Place the order
        self.tws.placeOrder(
                orderid,                                    # orderId,
                ibcontract,                                   # contract,
                iborder                                       # order
            )
    
        return orderid

    def any_open_orders(self):
        """
        Simple wrapper to tell us if we have any open orders
        """
        
        return len(self.get_open_orders())>0

    def get_open_orders(self):
        """
        Returns a list of any open orders
        """
        
        
        self.cb.init_openorders()
        self.cb.init_error()
                
        start_time=time.time()
        self.tws.reqAllOpenOrders()
        iserror=False
        finished=False
        
        while not finished and not iserror:
            finished=self.cb.flag_order_structure_finished
            iserror=self.cb.flag_iserror
            if (time.time() - start_time) > MAX_WAIT_SECONDS:
                ## You should have thought that IB would teldl you we had finished
                finished=True
            pass
        
        order_structure=self.cb.data_order_structure
        if iserror:
            print self.cb.error_msg
            print "Problem getting open orders"
    
        return order_structure    
    


    def get_executions(self, reqId=MEANINGLESS_NUMBER):
        """
        Returns a list of all executions done today
        """
        assert type(reqId) is int
        if reqId==FILL_CODE:
            raise Exception("Can't call get_executions with a reqId of %d as this is reserved for fills %d" % reqId)

        self.cb.init_fill_data()
        self.cb.init_error()
        
        ## We can change ExecutionFilter to subset different orders
        ef=ExecutionFilter();
        #ef.m_time="20160101"
        #ef.client_id=0;
        t=2;
       
        while t > 1:
            reqId=reqId+1
            self.tws.reqExecutions(reqId, ef)
    
            iserror=False
            finished=False
            
            start_time=time.time()
            
            while not finished and not iserror:
                finished=self.cb.flag_fill_data_finished
                iserror=self.cb.flag_iserror
                if (time.time() - start_time) > MAX_WAIT_SECONDS:
                    finished=True
                pass
        
            if iserror:
                print self.cb.error_msg
                print "Problem getting executions"
            
            t=t-1;
            
        execlist=self.cb.data_fill_data.values()
        
        return execlist
        
        
    def get_IB_account_data(self):

        time.sleep(2)
        self.cb.init_portfolio_data()
        self.cb.init_error()
        
        ## Turn on the streaming of accounting information
        
        self.tws.reqAccountUpdates(True, self.accountid)
        
        start_time=time.time()
        finished=False
        iserror=False

        while not finished and not iserror:
            finished=self.cb.flag_finished_portfolio
            iserror=self.cb.flag_iserror

            if (time.time() - start_time) > MAX_WAIT_SECONDS:
                finished=True
                print "Didn't get an end for account update, might be missing stuff"
            pass
        if iserror:
            print self.cb.error_msg
            print "Problem getting details"
            return None


        
        ## Turn off the streaming
        ## Note portfolio_structure will also be updated
        #self.tws.reqAccountUpdates(False, self.accountid)

        portfolio_data=self.cb.data_portfoliodata
        account_value=self.cb.data_accountvalue

        return (account_value, portfolio_data)


    def get_IB_market_data(self, ibcontract, tickerid=999):         
        """
        Returns granular market data
        
        Returns a tuple (bid price, bid size, ask price, ask size)
        
        """
        
        ## initialise the tuple
        self.cb.init_tickdata(tickerid)
        self.cb.init_error()
            
        # Request a market data stream 
        self.tws.reqMktData(
                tickerid,
                ibcontract,
                "",
                False)       
        
        start_time=time.time()

        #finished=False
        #iserror=False

        # while not finished and not iserror:
        #    iserror=self.cb.flag_iserror
        #    if (time.time() - start_time) > seconds:
        #        finished=True
        #    #pass
        #    time.sleep(10)
        #self.tws.cancelMktData(tickerid)
        
        marketdata=self.cb.data_tickdata[tickerid]
        ## marketdata should now contain some interesting information
        ## Note in this implementation we overwrite the contents with each tick; we could keep them
        
        #if iserror:
        #    print "Error: "+self.cb.error_msg
        #    print "Failed to get any prices with marketdata"
        
        return marketdata

    def get_realtimebar(self, ibcontract, ibtype, tickerid, data, filename):
        
        """
        Returns a list of snapshotted prices, averaged over 'real time bars'
        
        tws is a result of calling IBConnector()
        
        """
        
        tws=self.tws
        
        global finished
        global iserror
        global pricevalue
        global rtbar
        global rtdict
        global rtfile
        iserror=False
        
        finished=False
        pricevalue=[]
        rtdict[tickerid]=ibcontract.symbol + ibcontract.currency
        rtfile[tickerid]=filename
        rtbar[tickerid]=data
        # Request current price in 5 second increments
        # It turns out this is the only way to do it (can't get any other increments)
        
        tws.reqRealTimeBars(
                tickerid,                                          # tickerId,
                ibcontract,                                   # contract,
                5, 
                ibtype,
                0)
    
    
        start_time=time.time()
        ## get about 16 seconds worth of samples
        ## could obviously just stop at N bars as well eg. while len(pricevalue)<N:
        

        
        ## Cancel the stream
        #tws.cancelRealTimeBars(MEANINGLESS_ID)

        #if len(pricevalue)==0 or iserror:
        #    raise Exception("Failed to get price")

        
        return pricevalue
        
    def getDataFromIB(self, brokerData,endDateTime,data):
        WAIT_TIME=60
        global rtbar
        global rtdict
        global rthist
        iserror=False
        
        pricevalue=[]
        tickerid=brokerData['tickerId']
        rtdict[tickerid]=brokerData['symbol'] + brokerData['currency']
        rtbar[tickerid]=data
        #data_cons = pd.DataFrame()
        # Instantiate our callback object
        
    
        # Instantiate a socket object, allowing us to call TWS directly. Pass our
        # callback object so TWS can respond.
        tws = self.tws
        #tws = EPosixClientSocket(callback, reconnect_auto=True)
        # Connect to tws running on localhost
        
        # Simple contract for GOOG
        contract = Contract()
        contract.exchange = brokerData['exchange']
        contract.symbol = brokerData['symbol']
        contract.secType = brokerData['secType']
        contract.currency = brokerData['currency']
        ticker = contract.symbol+contract.currency
        #today = dt.today()
    
        print("\nRequesting historical data for %s" % ticker)
    
        # Request some historical data.
        rthist[tickerid]=Event()
        #for endDateTime in getHistLoop:
        tws.reqHistoricalData(
            brokerData['tickerId'],                                         # tickerId,
            contract,                                   # contract,
            endDateTime,                            #endDateTime
            brokerData['durationStr'],                                      # durationStr,
            brokerData['barSizeSetting'],                                    # barSizeSetting,
            brokerData['whatToShow'],                                   # whatToShow,
            brokerData['useRTH'],                                          # useRTH,
            brokerData['formatDate']                                          # formatDate
            )
    
    
        print("====================================================================")
        print(" %s History requested, waiting %ds for TWS responses" % (endDateTime, WAIT_TIME))
        print("====================================================================")
        
        try:
            rthist[tickerid].wait(timeout=WAIT_TIME)
        except KeyboardInterrupt:
            pass
        finally:
            if not rthist[tickerid].is_set():
                print('Failed to get history within %d seconds' % WAIT_TIME)
        
        #data_cons = pd.concat([data_cons,callback.data],axis=0)
                 
       
        return rtbar[tickerid]