from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

with open('\logs\post_processing_error_'+fulltimestamp+'.txt', 'w') as e:
    proc = Popen(['python', 'check_systems_live_ib.py','1','1','1','0'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'create_board_history.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'excel_charts.py','1'], stderr=e)
    proc.wait()
