#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:54:31 2017

@author: hidemiasakura
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from slackclient import SlackClient
from c2api.c2api import setDesiredPositions, get_working_signals, clear_signals,\
                        retrieveSystemEquity
from web.tsdp.betting.start_moc import restart_webserver
#from web.tsdp.betting.helpers import get_logfiles
c2id = "107146997"
c2key = "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w"
BOT_NAME = 'nsz'

#slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
SLACK_BOT_TOKEN='xoxb-159267896464-7paxh6FQWGz5x82qpCYNlqcs'
BOT_ID='U4P7VSCDN'
slack_client = SlackClient(SLACK_BOT_TOKEN)

def get_logfiles(search_string='', exclude=False):
    search_dir='/logs/'
    os.chdir(search_dir)
    files = filter(os.path.isfile, os.listdir(search_dir))
    if exclude:
        files = [os.path.join(search_dir, f) for f in files if search_string not in f]  # add path to each file
    else:
        files = [os.path.join(search_dir, f) for f in files if search_string in f]  # add path to each file

    files.sort(key=lambda x: os.path.getmtime(x))
    return files

def handle_command(command, channel):
    print command
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands. If not,
        returns back what it needs for clarification.
    """
    response = "Not sure what you mean."
    #response += " Use the *" + EXAMPLE_COMMAND + "* command for help."
    if command.startswith('ping'):
        response = "<@"+command.split()[1]+"> yo yo yo"
    if command.startswith('help'):
        response = "*commands:*\n"
        response += "BUY AAPL 100\n"
        response += "SELL AAPL 100\n"
        response += "CLOSE AAPL\n"
        #response += "CLEAR ORDERS\n"
        response += "GET OPEN ORDERS\n"
        response += "RESULTS\n"
        response += "RESTART SERVER\n"
        #response += "WHERE MY BABY DOLLS FRANK? :dancers:"

    if command.startswith('restart server'):
        restart_webserver()
        files=get_logfiles(search_string='restart_webserver_error')
        filename=files[-1]
        print filename
        response=filename+'\n'
        time.sleep(2)
        with open(filename, 'r') as f:
            for line in f: response+=line

    if command.startswith('buy'):
        symbol=command.split()[1]
        qty = command.split()[2]
        response = "BOUGHT " + str(qty) + " " + symbol
        orders = [{
                "symbol"		: symbol,
                "typeofsymbol"	: "stock",
                "quant"			: qty
             }]
        setDesiredPositions(orders)

    if command.startswith('sell'):
        symbol=command.split()[1]
        qty = int(command.split()[2])
        response = "SOLD " + str(qty) + " " + symbol
        orders = [{
                "symbol"		: symbol,
                "typeofsymbol"	: "stock",
                "quant"			: '-'+qty
             }]
        setDesiredPositions(orders)

    if command.startswith('close'):
        symbol=command.split()[1]
        #qty = int(command.split()[2])
        response = "CLOSED " + symbol
        orders = [{
                "symbol"		: symbol,
                "typeofsymbol"	: "stock",
                "quant"			: '0'
             }]
        setDesiredPositions(orders)

    #if command.startswith('clear orders'):
    #    response = "<@" + command.split()[1] + "> CLEARING ORDERS...\n"
    #    response += clear_signals(c2id, c2key)

    if command.startswith('get open orders'):
        response = "GETTING OPEN ORDERS...\n"
        response += get_working_signals(c2id, c2key)
        
    if command.startswith('results'):
        response = "RESULTS...\n"
        df = retrieveSystemEquity(c2id, c2key).groupby(['YYYYMMDD']).last()
        pc=round(df.strategy_with_cost.astype(float).pct_change()[-1]*100,2)
        benchmark_pc=round(df.index_price.astype(float).pct_change()[-1]*100)
        var_pc = pc-benchmark_pc
        itd = round(df.strategy_with_cost.astype(float).pct_change(periods=df.shape[0]-1)[-1]*100)
        benchmark_itd = round(df.index_price.astype(float).pct_change(periods=df.shape[0]-1)[-1]*100)
        var_itd=itd-benchmark_itd
        response+="Last Day: Frank: {}%   S&P500: {}%  VS: {}%\n".format(pc, benchmark_pc, var_pc)
        response+="ITD: Frank: {}%   S&P500: {}%  VS: {}%\n".format(itd, benchmark_itd, var_itd)
        
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)


def parse_slack_output(slack_rtm_output):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']\
                    and output['user'] in operators:
                print output_list
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']
    return None, None

# constants
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "help"

if __name__ == "__main__":
    api_call = slack_client.api_call("users.list")
    members = api_call.get('members')
    users=[(x.get('name'), x.get('id')) for x in members if 'name' in x] 
    print users
    operators=['U0D2R4FC5']
    #operators+=['U0D2TE3PC'] richard
    
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print(BOT_NAME+" connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                print channel, command
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
        
'''
if __name__ == "__main__":
    api_call = slack_client.api_call("users.list")
    if api_call.get('ok'):
        # retrieve all users so we can find our bot
        users = api_call.get('members')
        for user in users:
            if 'name' in user and user.get('name') == BOT_NAME:
                print("Bot ID for '" + user['name'] + "' is " + user.get('id'))
    else:
        print("could not find bot user with the name " + BOT_NAME)
'''