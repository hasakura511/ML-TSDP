cd \cygwin64\media\sf_python\tsdp
\anaconda2\python get_ibpos.py

\anaconda2\python v2_get_all_pairs.py
\anaconda2\python make_counter.py

start /b .\start_eurjpy.cmd
start /b .\start_eurusd.cmd
start /b .\start_gbpusd.cmd
start /b .\start_usdjpy.cmd
start /b .\start_usdchf.cmd
start /b .\start_audusd.cmd
start /b .\start_usdcad.cmd

sleep 500

\anaconda2\python system_EURJPY.py 1
\anaconda2\python system_EURUSD.py 1
\anaconda2\python system_GBPUSD.py 1
\anaconda2\python system_USDJPY.py 1
\anaconda2\python system_USDCHF.py 1
\anaconda2\python system_AUDUSD.py 1
\anaconda2\python system_USDCAD.py 1
\anaconda2\python proc_signal_v2.py

sleep 10
\anaconda2\python proc_pos_adj.py	

sleep 10
\anaconda2\python proc_signal_v2dps.py
\anaconda2\python system_BTCUSD_bitstamp_hourly.py 1
\anaconda2\python system_BTCUSD_bitstamp.py 1
