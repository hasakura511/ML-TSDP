cd \cygwin64\media\sf_python\tsdp

\anaconda2\python make_counter.py

rem start /b .\start_eurjpy_v3.cmd
rem start /b .\start_eurusd_v3.cmd
rem start /b .\start_gbpusd_v3.cmd
rem start /b .\start_usdjpy_v3.cmd
rem start /b .\start_usdchf_v3.cmd
rem start /b .\start_audusd_v3.cmd
rem start /b .\start_usdcad_v3.cmd

start /b .\start_eurjpy_v2.cmd
start /b .\start_eurusd_v2.cmd
start /b .\start_gbpusd_v2.cmd
start /b .\start_usdjpy_v2.cmd
start /b .\start_usdchf_v2.cmd
start /b .\start_audusd_v2.cmd
start /b .\start_usdcad_v2.cmd

PING -n 600 127.0.0.1>nul

\anaconda2\python get_ibpnl.py
\anaconda2\python system_BTCUSD_bitstamp_hourly.py 1
\anaconda2\python system_BTCUSD_bitstamp.py 1
