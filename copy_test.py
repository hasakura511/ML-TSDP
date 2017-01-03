from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
import shutil

shutil.copy('/ml-tsdp/suztoolz/check_systems_live_func.py','/ml-tsdp/check_systems_live_func.py')
#proc = Popen('copy /ml-tsdp/suztoolz/check_systems_live_func.py /ml-tsdp/', shell=True)