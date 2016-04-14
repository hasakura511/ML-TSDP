import quantiacsToolbox as qtb
import logging
logging.basicConfig(filename='/logs/qtb_s101.log',level=logging.DEBUG)

returnDict=qtb.runts('s101.py')
print returnDict
