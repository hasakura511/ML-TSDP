from shutil import copyfile
from datetime import datetime as dt
import time
from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
    
start_time = time.time()
src='./data/futures.sqlite3'
dst='Z:/tsdp/futures.sqlite3'
copyfile(src, dst)
print 'copied', src,'to',dst
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()