from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

import os
import shutil
src='\ml-tsdp\data\csidata\\v4futures2\\'
dest='\ml-tsdp\data\csidata\\v4futures4\\'
src_files = [x for x in os.listdir(src) if '.csv' in x.lower()]
i=0
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        i+=1
        shutil.copy(full_file_name, dest)
        print 'copied', full_file_name, 'to', dest
print len(src_files), 'source files', i, 'files copied'

with open('\logs\post_processing_csi_error_'+fulltimestamp+'.txt', 'w') as e:
    #f.flush()
    #e.flush()
    proc = Popen(['python', 'vol_adjsize_c2.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'vol_adjsize.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'refresh_c2.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'slip_report_c2.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'slip_report_ib.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'heatmap_futuresCSI.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'run_allsystems.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'vol_adjsize.py','1'], stderr=e)
    proc.wait()
