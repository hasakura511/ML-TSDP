set folder="\ML-TSDP\data\csidata\v4futures4"
cd /d %folder%
rem for /F "delims=" %%i in ('dir /b') do (del "%%i" /s/q)
del . /F /Q


cd \ML-TSDP\
\anaconda2\python create_board_history.py 1 >> \logs\daily_futures.log
rem \anaconda2\python vol_adjsize.py 1 >> \logs\daily_data.log