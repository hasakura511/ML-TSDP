from subprocess import Popen, PIPE, check_output, STDOUT
import datetime

def start_immediate():
    fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    with open('\logs\moc_live_error_'+fulltimestamp+'.txt', 'w') as e:
        #f.flush()
        #e.flush()
        proc = Popen(['/anaconda2/python', '/ml-tsdp/moc_live.py','1','1','1','1'], cwd='/ml-tsdp/',stderr=e)
        proc.wait()