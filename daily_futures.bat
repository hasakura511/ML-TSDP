cd \cygwin64\media\sf_python\tsdp
rem \anaconda2\python vol_adjsize.py 1 >> \logs\daily_futures.log
\anaconda2\python create_signalPlots.py 1 >> \logs\daily_futures.log
\anaconda2\python instRanking.py 1 >> \logs\daily_futures.log
rem \anaconda2\python proc_sig_adj.py >> \logs\daily_futures_orders.log
copy .\data\systems\system.csv .\data\systems\system_backup.csv
copy .\data\systems\system_v4mini.csv .\data\systems\system.csv
\anaconda2\python proc_signal_v4.py 1 v4mini >> \logs\daily_futures_orders.log
copy .\data\systems\system_backup.csv .\data\systems\system.csv
\anaconda2\python heatmap_futuresCSI.py 1 >> \logs\daily_futures.log
