cd \ML-TSDP\
\anaconda2\python post_processing.py >> \logs\daily_futures.log

cd \ml-tsdp\web\tsdp\
wmic process where "Commandline like '%%manage.py runserver%%' and name like '%%python.exe%%'" call terminate
\anaconda2\python runserver.py