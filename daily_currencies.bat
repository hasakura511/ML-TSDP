cd \cygwin64\media\sf_python\tsdp
\anaconda2\python proc_signal_v2.py 1 v4currencies >> \logs\daily_currencies.log
\anaconda2\python vol_adjsize_c.py 1 >> \logs\daily_currencies.log
\anaconda2\python heatmap_currencies2.py 1 >> \logs\daily_currencies.log
\anaconda2\python create_signalPlots_FX.py 1 >> \logs\daily_currencies.log
