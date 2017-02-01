from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
import shutil


def run_checksystems():
    fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    shutil.copy('/ml-tsdp/suztoolz/check_systems_live_func.py', '/ml-tsdp/check_systems_live_func.py')
    with open('\logs\checksystems_live_' + fulltimestamp + '.txt', 'w') as f:
        with open('\logs\checksystems_live_error_'+fulltimestamp+'.txt', 'w') as e:
            print('processing Check Systems...')
            proc = Popen(['/anaconda2/python', '/ml-tsdp/check_systems_live_ib.py','1'],\
                         cwd='/ml-tsdp/',stdout=f, stderr=e)
            proc.wait()
            f.flush()
            e.flush()
            proc = Popen(['/anaconda2/python', '/ml-tsdp/check_systems_live_func.py','1'],\
                         cwd='/ml-tsdp/',stdout=f, stderr=e)
            #proc.wait()
            

def get_newtimetable():
    fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    with open('\logs\get_timetable_' + fulltimestamp + '.txt', 'w') as f:
        with open('\logs\get_timetable_error_'+fulltimestamp+'.txt', 'w') as e:
            #f.flush()
            #e.flush()
            print('Attempting to get new timetable...')
            proc = Popen(['/anaconda2/python', '/ml-tsdp/moc_live.py','0','0','0','0'],\
                         cwd='/ml-tsdp/',stdout=f, stderr=e)
            #proc.wait()


if __name__ == "__main__":
    get_newtimetable()