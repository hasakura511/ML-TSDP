from subprocess import Popen, PIPE, check_output

with open('\logs\moc_live.txt', 'w') as f:
    with open('\logs\moc_live_error.txt', 'w') as e:
        proc = Popen(['python', '\cygwin64\media\sf_python\tsdp\moc_live.py','1'], stdout=f, stderr=e)
        proc.wait()