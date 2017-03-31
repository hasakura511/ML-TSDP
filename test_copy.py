from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

import os
import shutil
src='\ml-tsdp\data\csidata\\v4futures2\\'
check='\ml-tsdp\data\csidata\\v4futures4\\'
dest='\ml-tsdp\data\csidata\\v4futures5\\'
#check_files= [x for x in os.listdir(check) if '.csv' in x.lower()]
copy_files = [x for x in os.listdir(check) if '.csv' in x.lower()]


i=0
for file_name in copy_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        i+=1
        shutil.copy(full_file_name, dest)
        print 'copied', full_file_name, 'to', dest
print len(copy_files), 'source files', i, 'files copied'
