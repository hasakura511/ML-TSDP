#!/bin/sh
python get_ibpos.py
python system_EURJPY.py
python system_EURUSD.py
python system_GBPUSD.py
python system_USDJPY.py
python system_USDCHF.py
python system_AUDUSD.py
python system_USDCAD.py
python proc_signal_v2.py
python get_exec.py

python system_AUDUSD_v2.01.py 1
python proc_signal_v2dps.py
python get_exec.py
~                                                
