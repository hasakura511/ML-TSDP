from subprocess import Popen, PIPE, check_output, STDOUT


with open('\logs\moc_live_error.txt', 'a') as e:
    #f.flush()
    #e.flush()
    proc = Popen(['python', '\cygwin64\media\sf_python\\tsdp\moc_live.py','1','1','1','0'], stderr=e)
    proc.wait()