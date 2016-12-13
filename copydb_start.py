from shutil import copyfile
from datetime import datetime as dt
import time
from subprocess import Popen, PIPE, check_output, STDOUT, call
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

with open('\logs\copydb_error.txt', 'a') as e:
    #f.flush()
    #e.flush()



    proc = Popen(['python', '\cygwin64\media\sf_python\\tsdp\copydb.py'], stderr=e)
    proc.wait()
