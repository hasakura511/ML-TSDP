rem cd \cygwin64\media\sf_python\tsdp\data\tickerData
rem echo Y | del *.*
cd \cygwin64\media\sf_python\tsdp\data
rem \anaconda2\python rundata.py >> \logs\quantiacs_data.log
\anaconda2\python ratioAdjust.py >> \logs\quantiacs_data.log