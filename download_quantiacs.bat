cd \cygwin64\media\sf_python\tsdp\data
\anaconda2\python ratioAdjust2.py %1 >> \logs\quantiacs_data.log
cd \cygwin64\media\sf_python\tsdp
\anaconda2\python heatmap_futures.py 1 >> \logs\quantiacs_data.log
\anaconda2\python create_signalPlots.py 1 >> \logs\quantiacs_data.log