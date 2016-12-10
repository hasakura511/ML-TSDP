@echo OFF


net use z: \\webserver /U:webserver\RQuxvmG7e5Bh

cd \cygwin64\media\sf_python\tsdp\data
copy /y futures.sqlite3 Z:\tsdp\ 

@echo %time%