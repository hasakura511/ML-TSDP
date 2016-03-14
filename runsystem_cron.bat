cd \cygwin64\media\sf_python\tsdp

\anaconda2\python make_counter.py

start /b .\start_eurjpy.cmd
start /b .\start_eurusd.cmd
start /b .\start_gbpusd.cmd
start /b .\start_usdjpy.cmd
start /b .\start_usdchf.cmd
start /b .\start_audusd.cmd
start /b .\start_usdcad.cmd

PING -n 600 127.0.0.1>nul

\anaconda2\python system_BTCUSD_bitstamp_hourly.py 1
\anaconda2\python system_BTCUSD_bitstamp.py 1
