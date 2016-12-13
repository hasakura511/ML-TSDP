import os
from shutil import copyfile
from datetime import datetime as dt
import time
from subprocess import Popen, PIPE, check_output, STDOUT, call
import datetime
    
start_time = time.time()
#networkPath ='\\webserver\'
domain_name = 'WEBSERVER'
user_name = 'Administrator'
password = 'RQuxvmG7e5Bh'
src='./data/futures.sqlite3'
dst='M:/tsdp/futures.sqlite3'
# Disconnect anything on M
os.system(r"net use m: \\\169.254.194.221\tsdpWEB\ /U:WEBSERVER\Administrator RQuxvmG7e5Bh")
print 'didsaf'
#call(r'net use m: /del', shell=True)

# Connect to shared drive, use drive letter M
#call(r'net use '+networkPath+' /U:'+user+' '+ password, shell=True)
copyfile(src, dst)
print 'copied', src,'to',dst
os.system(r"NET USE M: /DELETE")
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()