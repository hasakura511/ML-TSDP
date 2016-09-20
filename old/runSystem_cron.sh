#!/bin/sh
cd /media/sf_Python/TSDP/

/anaconda2/python/v2_get_all_pairs.py
/anaconda2/python make_counter.py
/anaconda2/python system_EURJPY_v2.2C.py 1 &
/anaconda2/python system_EURUSD_v2.2C.py 1 &
/anaconda2/python system_GBPUSD_v2.2C.py 1 &
/anaconda2/python system_USDJPY_v2.2C.py 1 &
/anaconda2/python system_USDCHF_v2.2C.py 1 &
/anaconda2/python system_AUDUSD_v2.2C.py 1 &
/anaconda2/python system_USDCAD_v2.2C.py 1 &

/anaconda2/python system_EURJPY.py 1
/anaconda2/python system_EURUSD.py 1
/anaconda2/python system_GBPUSD.py 1
/anaconda2/python system_USDJPY.py 1
/anaconda2/python system_USDCHF.py 1
/anaconda2/python system_AUDUSD.py 1
/anaconda2/python system_USDCAD.py 1
/anaconda2/python get_ibpos.py
/anaconda2/python proc_signal_v2.py

sleep 360
/anaconda2/python get_ibpos.py
/anaconda2/python proc_signal_v2dps.py

/anaconda2/python system_BTCUSD_bitstamp_hourly.py 1
/anaconda2/python system_BTCUSD_bitstamp.py 1