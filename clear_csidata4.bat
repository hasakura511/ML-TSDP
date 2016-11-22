set folder="C:\cygwin64\media\sf_python\tsdp\data\csidata\v4futures4"
cd /d %folder%
rem for /F "delims=" %%i in ('dir /b') do (del "%%i" /s/q)
del . /F /Q


cd \cygwin64\media\sf_python\tsdp
\anaconda2\python vol_adjsize_c2.py 1 >> \logs\daily_data.log
\anaconda2\python vol_adjsize.py 1 >> \logs\daily_data.log