#!/bin/sh
cd \media\sf_python\tsdp
/anaconda2/bin/python get_ibpos.py
/anaconda2/bin/python proc_signal_v2.py	
/anaconda2/bin/python proc_signal_v2dps.py
sleep 100
/Users/admin/anaconda/bin/python get_ibpos.py
