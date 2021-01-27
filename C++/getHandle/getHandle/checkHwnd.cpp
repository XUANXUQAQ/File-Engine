#include "pch.h"
#include <algorithm>
#include <string>
#include <tchar.h>
#include <TlHelp32.h>
#include <Windows.h>

std::wstring get_process_name_by_handle(HWND nlHandle);

bool is_search_bar_window(const HWND& hd)
{
    char title[200];
    GetWindowTextA(hd, title, 200);
    return strcmp(title, "File-Engine-SearchBar") == 0;
}

HWND getSearchBarHWND()
{
    return FindWindowExA(nullptr, nullptr, nullptr, "File-Engine-SearchBar");
}

bool is_explorer_window_low_cost(const HWND& hwnd)
{
    if (IsWindowEnabled(hwnd) && !IsIconic(hwnd))
    {
        char className[200];
        GetClassNameA(hwnd, className, 200);
        std::string WindowClassName(className);
        transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
        //使用检测窗口类名的方式更节省CPU资源
        return WindowClassName.find("cabinet") != std::string::npos;
    }
    return false;
}

//不要在长时间循环中使用
bool isExplorerWindowHighCost(const HWND& hwnd)
{
	std::wstring proc_name = get_process_name_by_handle(hwnd);
    transform(proc_name.begin(), proc_name.end(), proc_name.begin(), ::tolower);
    return proc_name.find(_T("explorer.exe")) != std::wstring::npos;
}

std::wstring get_process_name_by_handle(HWND nlHandle)
{
	std::wstring loStrRet;
    //得到该进程的进程id
    DWORD ldwProID;
    GetWindowThreadProcessId(nlHandle, &ldwProID);
    if (0 == ldwProID)
    {
        return L"";
    }
	auto* const handle = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (handle == reinterpret_cast<HANDLE>(-1))
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


bool is_file_chooser_window(const HWND& hwnd)
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
        //排除已知的例外
        (
            windowTitle.find("internet download manager") == std::string::npos &&
            windowTitle.find("push commits to") == std::string::npos &&
            windowTitle.find("geek uninstaller") == std::string::npos &&
            windowTitle.find("rainmeter") == std::string::npos &&
            windowTitle.find("techpowerup") == std::string::npos &&
            WindowClassName.find("sunawt") == std::string::npos
		);
}