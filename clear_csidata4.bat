set folder="C:\cygwin64\media\sf_python\tsdp\data\csidata\v4futures4"
cd /d %folder%
rem for /F "delims=" %%i in ('dir /b') do (del "%%i" /s/q)
del . /F /Q