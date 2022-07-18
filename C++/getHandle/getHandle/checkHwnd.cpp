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
 * ��鴰���ǲ���������
 */
bool is_search_bar_window(const HWND& hd)
{
	char title[200];
	GetWindowTextA(hd, title, 200);
	return strcmp(title, "File-Engine-SearchBar") == 0;
}

/**
 * ��ȡ�����򴰿ھ��
 */
HWND get_search_bar_hwnd()
{
	return FindWindowExA(nullptr, nullptr, nullptr, "File-Engine-SearchBar");
}

/**
 * ͨ�������ж��ǲ���explorer.exe����
 */
bool is_explorer_window_by_class_name(const HWND& hwnd)
{
	if (IsWindowEnabled(hwnd) && !IsIconic(hwnd))
	{
		char className[200];
		GetClassNameA(hwnd, className, 200);
		std::string WindowClassName(className);
		transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
		//ʹ�ü�ⴰ�������ķ�ʽ����ʡCPU��Դ
		if (WindowClassName.find("cabinet") != std::string::npos)
		{
			if (IsWindows8OrGreater())
			{
				return FindWindowExA(hwnd, nullptr, "UIRibbonCommandBarDock", nullptr);
			}
			// windows 7
			HWND tmp = FindWindowExA(hwnd, nullptr, "ShellTabWindowClass", nullptr);
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
 * ͨ���������ж��ǲ���explorer����
 */
bool is_explorer_window_by_process(const HWND& hwnd)
{
	std::wstring proc_name = get_process_name_by_handle(hwnd);
	transform(proc_name.begin(), proc_name.end(), proc_name.begin(), tolower);
	return proc_name.find(_T("explorer")) != std::wstring::npos;
}

/**
 * ͨ�����ھ����ȡ������Ϣ
 */
std::wstring get_process_name_by_handle(HWND nlHandle)
{
	std::wstring loStrRet;
	//�õ��ý��̵Ľ���id
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
 * �жϴ��ھ���Ƿ�Ϊ�ļ�ѡ�񴰿�
 */
bool is_file_chooser_window(const HWND& hwnd)
{
	char _windowClassName[100];
	char title[100];
	int hasToolbar = false;
	EnumChildWindows(hwnd, is_hwnd_has_toolbar, reinterpret_cast<LPARAM>(&hasToolbar));
	if (hasToolbar)
	{
		GetClassNameA(hwnd, _windowClassName, 100);
		GetWindowTextA(hwnd, title, 100);
		std::string windowTitle(title);
		std::string WindowClassName(_windowClassName);
		std::transform(windowTitle.begin(), windowTitle.end(), windowTitle.begin(), ::tolower);
		std::transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
		return WindowClassName.find("#32770") != std::string::npos ||
			WindowClassName.find("dialog") != std::string::npos;
	}
	return false;
}

/**
 * �жϴ��ھ���Ƿ�Ϊexplorer�Ϸ���toolbar
 */
BOOL CALLBACK is_hwnd_has_toolbar(HWND hwndChild, LPARAM lParam)
{
	char windowClassName[100] = {'\0'};
	GetClassNameA(hwndChild, windowClassName, 100);
	if (strcmp(windowClassName, "ToolbarWindow32") == 0)
	{
		*reinterpret_cast<int*>(lParam) = true;
		return false;
	}
	return true;
}
