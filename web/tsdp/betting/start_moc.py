from subprocess import Popen, PIPE, check_output, STDOUT
import datetime

def start_moc():
    fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    with open('\logs\moc_live_' + fulltimestamp + '.txt', 'w') as f:
        with open('\logs\moc_live_error_'+fulltimestamp+'.txt', 'w') as e:
            #f.flush()
            #e.flush()
            proc = Popen(['/anaconda2/python', '/ml-tsdp/moc_live.py','1','1','1','0'],\
                         cwd='/ml-tsdp/',stdout=f, stderr=e)
            #proc.wait()
            print('MOC processed.')