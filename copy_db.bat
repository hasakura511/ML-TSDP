@echo OFF
copy /y C:\cygwin64\media\sf_python\tsdp\data\futures.sqlite3 C:\cygwin64\media\sf_python\tsdp\data\results\
 
rem \anaconda2\python copydb_start.py
rem net use m: \\webserver\tsdp/U:WEBSERVER\Administrator RQuxvmG7e5Bh /persistent:no
rem start copy /y "C:\cygwin64\media\sf_python\tsdp\data\futures.sqlite3" "m:\"
rem pushd \\webserver\tsdp\
rem xcopy /y C:\cygwin64\media\sf_python\tsdp\data\futures.sqlite3 \\webserver\tsdp\
rem popd \\webserver\tsdp\
@echo %time%