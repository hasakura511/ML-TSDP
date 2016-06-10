cd \cygwin64\media\sf_python\tsdp
\anaconda2\python proc_sig_adj.py >> \logs\daily_futures.log
\anaconda2\python proc_signal_v2.py 1 v4futures >> \logs\daily_futures.log
\anaconda2\python vol_adjsize.py 1 >> \logs\daily_futures.log
\anaconda2\python heatmap_futuresCSI.py 1 >> \logs\daily_futures.log
\anaconda2\python create_signalPlots.py 1 >> \logs\daily_futures.log