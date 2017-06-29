from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

with open('\logs\post_processing_error_'+fulltimestamp+'.txt', 'w') as e:
	#post_processing flag is set to true, which checks the ib positions and submits orders
	#this causes problems when user changes orders after the MOC.
    #proc = Popen(['python', 'moc_live.py','0','0','1','1'], stderr=e)
    #proc.wait()
    #e.flush()
    proc = Popen(['python', 'create_board_history.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'excel_charts.py','1'], stderr=e)
    proc.wait()
