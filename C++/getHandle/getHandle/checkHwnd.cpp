#include "pch.h"
#include <algorithm>
#include <string>
#include <tchar.h>
#include <tlhelp32.h>
#include <Windows.h>

std::wstring GetProcessNameByHandle(HWND nlHandle);

bool isSearchBarWindow(const HWND& hd)
{
    char title[200];
    GetWindowTextA(hd, title, 200);
    return strcmp(title, "File-Engine-SearchBar") == 0;
}

bool isExplorerWindowLowCost(const HWND& hwnd)
{
    if (IsWindowEnabled(hwnd) && !IsIconic(hwnd))
    {
        char className[200];
        GetClassNameA(hwnd, className, 200);
        std::string WindowClassName(className);
        transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
        //ʹ�ü�ⴰ�������ķ�ʽ����ʡCPU��Դ
        return WindowClassName.find("cabinet") != std::string::npos;
    }
    return false;
}

//��Ҫ�ڳ�ʱ��ѭ����ʹ��
bool isExplorerWindowHighCost(const HWND& hwnd)
{
	std::wstring proc_name = GetProcessNameByHandle(hwnd);
    transform(proc_name.begin(), proc_name.end(), proc_name.begin(), ::tolower);
    return proc_name.find(_T("explorer.exe")) != std::wstring::npos;
}

std::wstring GetProcessNameByHandle(HWND nlHandle)
{
	std::wstring loStrRet;
    //�õ��ý��̵Ľ���id
    DWORD ldwProID;
    GetWindowThreadProcessId(nlHandle, &ldwProID);
    if (0 == ldwProID)
    {
        return L"";
    }
    HANDLE handle = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (handle == (HANDLE)-1)
    {
        //AfxMessageBox(L"CreateToolhelp32Snapshot error");
        return loStrRet;
    }
    PROCESSENTRY32 procinfo;
    procinfo.dwSize = sizeof(PROCESSENTRY32);
    BOOL more = ::Process32First(handle, &procinfo);
    while (more)
    {
        if (procinfo.th32ProcessID == ldwProID)
        {
            loStrRet = procinfo.szExeFile;
            CloseHandle(handle);
            return loStrRet;
        }
        more = Process32Next(handle, &procinfo);
    }
    CloseHandle(handle);
    return loStrRet;
}


bool isFileChooserWindow(const HWND& hwnd)
{
    char _windowClassName[100];
    char title[100];
    GetClassNameA(hwnd, _windowClassName, 100);
    GetWindowTextA(hwnd, title, 100);
    std::string windowTitle(title);
    std::string WindowClassName(_windowClassName);
    std::transform(windowTitle.begin(), windowTitle.end(), windowTitle.begin(), ::tolower);
    std::transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
    return ((WindowClassName.find("#32770") != std::string::npos ||
        WindowClassName.find("dialog") != std::string::npos))
        &&
        //�ų���֪������
        (
            windowTitle.find("internet download manager") == std::string::npos &&
            windowTitle.find("push commits to") == std::string::npos &&
            windowTitle.find("geek uninstaller") == std::string::npos &&
            windowTitle.find("rainmeter") == std::string::npos &&
            windowTitle.find("techpowerup") == std::string::npos &&
            WindowClassName.find("sunawt") == std::string::npos
		);
}