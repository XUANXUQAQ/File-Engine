#include <iostream>
#include <windows.h>
#include <tlhelp32.h>
#include <tchar.h>

using namespace std;

string GetExePath(void)
{
    char szFilePath[MAX_PATH + 1] = {0};
    GetModuleFileNameA(NULL, szFilePath, MAX_PATH);
    (strrchr(szFilePath, '\\'))[0] = 0; // 删除文件名，只获得路径字串
    string path = szFilePath;

    return path;
}

BOOL IsExistProcess(const char *pName)
{
	HANDLE hProcessSnap;
	PROCESSENTRY32 pe32;
	DWORD dwPriorityClass;

	bool bFind = false;
	// Take a snapshot of all processes in the system.
	hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hProcessSnap == INVALID_HANDLE_VALUE)
	{
		return false;
	}

	// Set the size of the structure before using it.
	pe32.dwSize = sizeof(PROCESSENTRY32);

	// Retrieve information about the first process,
	// and exit if unsuccessful
	if (!Process32First(hProcessSnap, &pe32))
	{
		CloseHandle(hProcessSnap); // clean the snapshot object
		return false;
	}

	// Now walk the snapshot of processes, and
	// display information about each process in turn
	do
	{
		// Retrieve the priority class.
		dwPriorityClass = 0;
		if (::strstr((const char *)pe32.szExeFile, pName) != NULL)
		{
			bFind = true;
			break;
		}
	} while (Process32Next(hProcessSnap, &pe32));

	CloseHandle(hProcessSnap);
	return bFind;
}

int main(int argc, char *argv[])
{
	if (argc == 3)
	{
		char path[500];
		memset(path, 0, 500);
		strcpy(path, argv[1]);
		char name[50];
		memset(name, 0, 50);
		strcpy(name, argv[2]);
		int count = 0;
        cout << "file path:" << path << endl;
		while (true)
		{
			Sleep(50);
			if (count > 20)
			{
				cout << "over time" << endl;
				return 0; //超时
			}
			if (!IsExistProcess(name)){
				break;
			}
		}
		ShellExecute(NULL, _T("open"), (LPCSTR)path, NULL, NULL, SW_SHOW);
	}
}