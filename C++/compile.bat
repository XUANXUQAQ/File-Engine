g++ -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o fileSearcher64.exe fileSearcher.cpp
g++ -m32 -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o fileSearcher86.exe fileSearcher.cpp
g++ -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o fileMonitor64.dll fileMonitor.cpp
g++ -m32 -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o fileMonitor86.dll fileMonitor.cpp
g++ -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o getAscII64.dll getAscII.cpp
g++ -m32 -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o getAscII86.dll getAscII.cpp
g++ -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o hotkeyListener64.dll hotkeyListener.cpp
g++ -m32 -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o hotkeyListener86.dll hotkeyListener.cpp
g++ -m32 -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o updater86.exe updater.cpp
g++ -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o updater64.exe updater.cpp
g++ -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o restart64.exe restart.cpp
g++ -m32 -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o restart86.exe restart.cpp
g++ -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o isLocalDisk64.dll isLocalDisk.cpp
g++ -m32 -shared -finput-charset=UTF-8 -fexec-charset=UTF-8 -O3 -o isLocalDisk86.dll isLocalDisk.cpp