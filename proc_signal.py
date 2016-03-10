import numpy as np
import pandas as pd
import subprocess

subprocess.call(['python', 'get_ibpos.py'])
subprocess.call(['python', 'proc_signal_v2.py'])
subprocess.call(['python', 'get_ibpos.py'])
subprocess.call(['python', 'proc_signal_v2dps.py'])
