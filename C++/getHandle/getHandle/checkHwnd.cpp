#include "pch.h"
#include <algorithm>
#include <string>
#include <tchar.h>
#include <TlHelp32.h>
#include <Windows.h>
#include <VersionHelpers.h>

std::wstring get_process_name_by_handle(HWND nlHandle);
BOOL CALLBACK is_hwnd_has_toolbar(HWND hwndChild, LPARAM lParam);

/**
 * 检查窗口是不是搜索框
 */
bool is_search_bar_window(const HWND& hd)
{
	char title[200];
	GetWindowTextA(hd, title, 200);
	return strcmp(title, "File-Engine-SearchBar") == 0;
}

/**
 * 获取搜索框窗口句柄
 */
HWND get_search_bar_hwnd()
{
	return FindWindowExA(nullptr, nullptr, nullptr, "File-Engine-SearchBar");
}

/**
 * 通过类名判断是不是explorer.exe窗口
 */
bool is_explorer_window_by_class_name(const HWND& hwnd)
{
	if (IsWindowEnabled(hwnd) && !IsIconic(hwnd))
	{
		char class_name[200];
		GetClassNameA(hwnd, class_name, sizeof class_name);
		std::string window_class_name(class_name);
		std::transform(window_class_name.begin(), window_class_name.end(), window_class_name.begin(), std::tolower);
		//使用检测窗口类名的方式更节省CPU资源
		if (window_class_name.find("cabinet") != std::string::npos)
		{
			if (IsWindows8OrGreater())
			{
				if (FindWindowExA(hwnd, nullptr, "UIRibbonCommandBarDock", nullptr))
				{
					return true;
				}
				BOOL CALLBACK is_hwnd_has_cabinetWClass(HWND hwndChild, LPARAM lParam);
				int has_cabinet = 0;
				EnumChildWindows(hwnd, is_hwnd_has_cabinetWClass, reinterpret_cast<LPARAM>(&has_cabinet));
				return has_cabinet == 7;
			}
			// windows 7
			const HWND tmp = FindWindowExA(hwnd, nullptr, "ShellTabWindowClass", nullptr);
			if (tmp)
			{
				return !FindWindowExA(
					tmp,
					nullptr,
					"SHELLDLL_DefView",
					nullptr);
			}
		}
	}
	return false;
}

/**
 * 通过进程名判断是不是explorer窗口
 */
bool is_explorer_window_by_process(const HWND& hwnd)
{
	std::wstring proc_name = get_process_name_by_handle(hwnd);
	transform(proc_name.begin(), proc_name.end(), proc_name.begin(), tolower);
	return proc_name.find(_T("explorer")) != std::wstring::npos;
}

/**
 * 通过窗口句柄获取进程信息
 */
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


/**
 * 判断窗口句柄是否为文件选择窗口
 */
bool is_file_chooser_window(const HWND& hwnd)
{
	int has_toolbar = false;
	EnumChildWindows(hwnd, is_hwnd_has_toolbar, reinterpret_cast<LPARAM>(&has_toolbar));
	if (has_toolbar)
	{
		char title[100];
		char window_class_name[100];
		GetClassNameA(hwnd, window_class_name, 100);
		GetWindowTextA(hwnd, title, 100);
		std::string windowTitle(title);
		std::string WindowClassName(window_class_name);
		std::transform(windowTitle.begin(), windowTitle.end(), windowTitle.begin(), ::tolower);
		std::transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
		return WindowClassName.find("#32770") != std::string::npos ||
			WindowClassName.find("dialog") != std::string::npos;
	}
	return false;
}

/**
 * 判断窗口句柄是否为explorer上方的toolbar
 */
BOOL CALLBACK is_hwnd_has_toolbar(HWND hwndChild, LPARAM lParam)
{
	char window_class_name[100] = {'\0'};
	GetClassNameA(hwndChild, window_class_name, sizeof window_class_name);
	if (strcmp(window_class_name, "ToolbarWindow32") == 0)
	{
		*reinterpret_cast<int*>(lParam) = true;
		return false;
	}
	return true;
}

/**
 * 判断窗口句柄是否为explorer上方的toolbar
 */
BOOL CALLBACK is_hwnd_has_cabinetWClass(HWND hwndChild, LPARAM lParam)
{
	const auto flag = reinterpret_cast<int*>(lParam);
	char window_class_name[100] = { '\0' };
	GetClassNameA(hwndChild, window_class_name, sizeof window_class_name);
	const std::string class_name_str(window_class_name);
	if (class_name_str.find("Windows.UI.Core.CoreWindow") != std::string::npos)
	{
		*flag = 1;
	}
	else if (*flag & 1 && class_name_str.find("WorkerW") != std::string::npos)
	{
		*flag |= 2;
	}
	else if (*flag & 2 && class_name_str.find("ToolbarWindow32") != std::string::npos)
	{
		*flag |= 4;
	}
	if (*flag == 7)
	{
		return false;
	}
	return true;
}
