cd \cygwin64\media\sf_python\tsdp
del data\from_mt4\LOG*.*
rm data/from_mt4/*/LOG*
del c:\mql4\files\LOG*.*
rm /cygdrive/c/mql4/files/*/LOG*
xcopy /y /c /z data\signals\*.* C:\mql4\files\signals\
xcopy /y /c /z C:\mql4\files\bars\*.* data\bars\
xcopy /y /c /z C:\mql4\files\bidask\*.* data\bidask\
xcopy /y /c /z C:\mql4\files\*.* data\from_mt4\
xcopy /y /c /z C:\mql4\files\fut\*.* data\from_mt4\fut\
rem xcopy /y /c /z C:\mql4\files\usstocks\*.* data\from_mt4\usstocks\
exit
