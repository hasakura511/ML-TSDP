cd \cygwin64\media\sf_python\tsdp
del data\from_mt4\LOG*.*
rm data/from_mt4/*/LOG*
del c:\mql4\files\LOG*.*
rm /cygdrive/c/mql4/files/*/LOG*
xcopy /y /s /c /z C:\mql4\files\*.* data\from_mt4\
xcopy /y /s /c /z C:\mql4\files\bars\*.* data\from_mt4\bars\
xcopy /y /s /c /z C:\mql4\files\bidask\*.* data\from_mt4\bidask\
exit
