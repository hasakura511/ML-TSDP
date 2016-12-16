cd \ml-tsdp
\anaconda2\python vol_adjsize_c2.py 1 >> \logs\daily_data.log
\anaconda2\python vol_adjsize.py 1 >> \logs\daily_data.log
\anaconda2\python refresh_c2.py >> \logs\daily_futures.log
\anaconda2\python slip_report2.py 1 >> \logs\daily_data.log
\anaconda2\python slip_report_ib.py 1 >> \logs\daily_data.log
\anaconda2\python heatmap_futuresCSI.py 1 >> \logs\daily_data.log