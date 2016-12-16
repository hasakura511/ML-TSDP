from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

with open('\logs\moc_live_error_'+fulltimestamp+'.txt', 'w') as e:
    #f.flush()
    #e.flush()
    proc = Popen(['python', 'moc_live.py','1','1','1','0'], stderr=e)
    proc.wait()