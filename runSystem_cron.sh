#!/bin/sh
export PATH="/usr/local/bin:/usr/local/sbin:~/Documents/Android/sdk/platform-tools:~/Documents/Android/sdk/tools:~/Documents/Android/android-ndk-r10e:$PATH"

export LDFLAGS="-L/usr/local/opt/tcl-tk/lib:$LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/tcl-tk/include:$CPPFLAGS"
export TK_LIBRARY="/usr/local/Cellar/tcl-tk/8.6.4/lib"
export LD_LIBRARY_PATH="/usr/local/opt/tcl-tk/lib:$LD_LIBRARY_PATH"

export MONO_GAC_PREFIX="/usr/local"
#export DYLD_LIBRARY_PATH="/usr/local/Cellar/openssl/1.0.2d/lib"
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/anaconda/lib/:$DYLD_FALLBACK_LIBRARY_PATH"
# added by Anaconda 2.3.0 installer
export PATH="/Users/admin/anaconda/bin:$PATH"

# added by Anaconda 2.3.0 installer
export PATH="/Users/admin/anaconda/bin:$PATH"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

cd /Users/admin/ML-TSDP/
/Users/admin/anaconda/bin/python system_EURUSD.py 1
/Users/admin/anaconda/bin/python system_GBPUSD.py 1
/Users/admin/anaconda/bin/python system_USDJPY.py 1
/Users/admin/anaconda/bin/python system_USDCHF.py 1
/Users/admin/anaconda/bin/python system_AUDUSD.py 1
/Users/admin/anaconda/bin/python system_USDCAD.py 1
/Users/admin/anaconda/bin/python proc_signal.py
