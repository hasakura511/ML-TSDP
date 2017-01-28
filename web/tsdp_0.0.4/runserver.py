from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

#with open('/logs/runserver_' + fulltimestamp + '.txt', 'w') as f:
with open('/logs/runserver_'+fulltimestamp+'_error.txt', 'w') as e:
    #f.flush()
    #e.flush()
    proc = Popen(['python', 'manage.py','runserver','0.0.0.0:80','--noreload'], stderr=e)
    #proc.wait()
    