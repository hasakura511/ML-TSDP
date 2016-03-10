cd \cygwin64\media\sf_python\tsdp
\anaconda2\python get_ibpos.py
\anaconda2\python system_EURJPY.py 1
\anaconda2\python system_EURUSD.py 1
\anaconda2\python system_GBPUSD.py 1
\anaconda2\python system_USDJPY.py 1
\anaconda2\python system_USDCHF.py 1
\anaconda2\python system_AUDUSD.py 1
\anaconda2\python system_USDCAD.py 1
\anaconda2\python proc_signal_v2.py

\anaconda2\python get_ibpos.py

\anaconda2\python system_AUDUSD_v2.1.py 1
\anaconda2\python proc_signal_v2dps.py

sleep 10
\anaconda2\python get_ibpos.py
sleep 10
\anaconda2\python proc_pos_adj.py	

\anaconda2\python get_ibpos.py

\anaconda2\python system_BTCUSD_bitstamp_hourly.py 1
\anaconda2\python system_BTCUSD_bitstamp.py 1
rem \anaconda2\python get_exec.py
