set folder="\ML-TSDP\data\csidata\v4futures4"
cd /d %folder%
rem for /F "delims=" %%i in ('dir /b') do (del "%%i" /s/q)
del . /F /Q


rem cd \ML-TSDP\
rem \anaconda2\python vol_adjsize_c2.py 1 >> \logs\daily_data.log
rem \anaconda2\python vol_adjsize.py 1 >> \logs\daily_data.log