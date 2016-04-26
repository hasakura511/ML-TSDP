import subprocess
p = subprocess.Popen('/bin/bash',
                             shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             creationflags=0)
p.communicate()
