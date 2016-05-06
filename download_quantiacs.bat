cd \cygwin64\media\sf_python\tsdp\data\tickerData
echo Y | del *.*
cd \cygwin64\media\sf_python\tsdp\data
\anaconda2\python rundata.py >> \logs\quantiacs_data.log