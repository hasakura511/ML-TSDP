rem cd \cygwin64\media\sf_python\tsdp\data\tickerData
rem echo Y | del *.*
cd \cygwin64\media\sf_python\tsdp\data
rem \anaconda2\python rundata.py >> \logs\quantiacs_data.log
\anaconda2\python ratioAdjust2.py %1 >> \logs\quantiacs_data.log
cd \cygwin64\media\sf_python\tsdp
\anaconda2\python heatmap_futures.py 1 >> \logs\quantiacs_data.log
\anaconda2\python heatmap_currencies2.py 1 >> \logs\quantiacs_data.log