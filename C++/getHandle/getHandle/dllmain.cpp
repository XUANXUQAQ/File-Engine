// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <Windows.h>
#include <iostream>
#include <stdint.h>
#include <tlhelp32.h>
#include <tchar.h>
#include <thread>
#include <dwmapi.h>
#include <algorithm>
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "kernel32")
#define IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0)

constexpr auto EXPLORER_MIN_HEIGHT = 200;       //当窗口大小满足这些条件后才开始判断是否为explorer.exe
constexpr auto EXPLORER_MIN_WIDTH = 200;
constexpr auto EXPLORER_MIN_X_POS = 50;
constexpr auto EXPLORER_MIN_Y_POS = 50;
//#define TEST


using namespace std;

volatile bool isExplorerWindowAtTop = false;
volatile bool isRunning = false;
int explorerX;
int explorerY;
volatile long explorerWidth;
volatile long explorerHeight;
int toolbar_click_x;
int toolbar_click_y;
volatile bool is_click_not_explorer_or_searchbar;
HWND searchBarHWND;


void getTopWindow(HWND* hwnd);
void getWindowRect(const HWND& hwnd, LPRECT lprect);
bool isExplorerWindow(const HWND& hwnd);
void _start();
bool isFileChooserWindow(const HWND& hwnd);
void setClickPos(const HWND& fileChooserHwnd);
BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam);
BOOL CALLBACK findSeachBar(HWND hwndChild, LPARAM lParam);
bool isClickNotExplorerOrSearchBarOrSwitchTask();
void _checkMouseStatus();
bool isMouseClickedOrSwitchTaskPressed();
bool isSearchBarWindow(const HWND& hd);
void grabSearchBarHWND();
wstring GetProcessNameByHandle(HWND nlHandle);
extern "C" __declspec(dllexport) bool isDialogNotExist();
extern "C" __declspec(dllexport) void start();
extern "C" __declspec(dllexport) void stop();
extern "C" __declspec(dllexport) bool is_explorer_at_top();
extern "C" __declspec(dllexport) long getExplorerX();
extern "C" __declspec(dllexport) long getExplorerY();
extern "C" __declspec(dllexport) long getExplorerWidth();
extern "C" __declspec(dllexport) long getExplorerHeight();
extern "C" __declspec(dllexport) int get_toolbar_click_x();
extern "C" __declspec(dllexport) int get_toolbar_click_y(); 
extern "C" __declspec(dllexport) bool isExplorerAndSearchbarNotFocused();
extern "C" __declspec(dllexport) void resetMouseStatus();
extern "C" __declspec(dllexport) void transferSearchBarFocus();

__declspec(dllexport) bool isExplorerAndSearchbarNotFocused()
{
    return is_click_not_explorer_or_searchbar;
}

__declspec(dllexport) void transferSearchBarFocus()
{
    if (!searchBarHWND)
    {
        grabSearchBarHWND();
    }
    if (searchBarHWND)
    {
        //转移窗口焦点
        SendMessage(searchBarHWND, WM_KILLFOCUS, NULL, NULL);
    }
}

void grabSearchBarHWND()
{
    EnumWindows(findSeachBar, NULL);
}

bool isClickNotExplorerOrSearchBarOrSwitchTask()
{
    POINT point;
    BOOL ret;
    HWND hd;    //鼠标位置的窗口句柄
    if (isMouseClickedOrSwitchTaskPressed())
    {
        ret = GetCursorPos(&point);
        if (ret)
        {
            hd = WindowFromPoint(point);
            if (isExplorerWindow(hd) || isFileChooserWindow(hd))
            {
                return false;
            } 
            else
            {
                //检查是否点击搜索框
                return !(isSearchBarWindow(hd));
            }
        }
    }
    return false;
}

bool isSearchBarWindow(const HWND& hd)
{
    char title[200];
    GetWindowTextA(hd, title, 200);
    return strcmp(title, "File-Engine-SearchBar") == 0;
}

bool isMouseClickedOrSwitchTaskPressed()
{
    return IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_RBUTTON) || IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_MBUTTON) || IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_LBUTTON) ||
        (IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_MENU) && IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_TAB));
}

__declspec(dllexport) bool isDialogNotExist()
{
    HWND topWindow;
    getTopWindow(&topWindow);
    bool isFileChooserAtTop = isFileChooserWindow(topWindow);
    if (isFileChooserAtTop)
    {
        return false;
    }
    else
    {
        RECT windowRect;
        HWND hd = GetDesktopWindow();        //得到桌面窗口
        hd = GetWindow(hd, GW_CHILD);        //得到屏幕上第一个子窗口
        while (hd != NULL)                    //循环得到所有的子窗口
        {
            if (IsWindowVisible(hd) && !IsIconic(hd))
            {
                getWindowRect(hd, &windowRect);
                int tmp_explorerX = windowRect.left;
                int tmp_explorerY = windowRect.top;
                int tmp_explorerWidth = windowRect.right - windowRect.left;
                int tmp_explorerHeight = windowRect.bottom - windowRect.top;
                if (!(tmp_explorerHeight < EXPLORER_MIN_HEIGHT || tmp_explorerWidth < EXPLORER_MIN_WIDTH || tmp_explorerX < EXPLORER_MIN_X_POS || tmp_explorerY < EXPLORER_MIN_Y_POS))
                {
                    if (isExplorerWindow(hd))
                    {
                        return false;
                    }
                }
            }
            hd = GetWindow(hd, GW_HWNDNEXT);
        }
    }
    return true;
}

void getTopWindow(HWND* hwnd)
{
    *hwnd = ::GetForegroundWindow();
}

void getWindowRect(const HWND& hwnd, LPRECT lprect)
{
    GetWindowRect(hwnd, lprect);
}

bool isFileChooserWindow(const HWND& hwnd)
{
    char _windowClassName[100];
    char title[100];
    GetClassNameA(hwnd, _windowClassName, 100);
    GetWindowTextA(hwnd, title, 100);
    string windowTitle(title);
    string WindowClassName(_windowClassName);
    transform(windowTitle.begin(), windowTitle.end(), windowTitle.begin(), ::tolower);
    transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
    return ((WindowClassName.find("#32770") != string::npos ||
        WindowClassName.find("dialog") != string::npos)) && 
        windowTitle.find("internet download manager") == string::npos && 
        windowTitle.find("push commits to") == string::npos &&
        windowTitle.find("geek uninstaller") == string::npos;
}

void setClickPos(const HWND& fileChooserHwnd)
{
    EnumChildWindows(fileChooserHwnd, findToolbar, NULL);
}

BOOL CALLBACK findSeachBar(HWND hwndChild, LPARAM lParam)
{
    if (isSearchBarWindow(hwndChild))
    {
        return false;
    }
    return true;
}

BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam)
{
    HWND hwd2 = FindWindowExA(hwndChild, NULL, "Address Band Root", NULL);
    if (hwd2)
    {
        RECT rect;
        getWindowRect(hwd2, &rect);
        int toolbar_x = rect.left;
        int toolbar_y = rect.top;
        int toolbar_width = rect.right - rect.left;
        int toolbar_height = rect.bottom - rect.top;
        toolbar_click_x = toolbar_x + toolbar_width - 80;
        toolbar_click_y = toolbar_y + (toolbar_height / 2);
        return false;
    }
    return true;
}

wstring GetProcessNameByHandle(HWND nlHandle)
{
    wstring loStrRet = L"";
    //得到该进程的进程id
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

bool isExplorerWindow(const HWND& hwnd)
{
    if (IsWindowEnabled(hwnd) && !IsIconic(hwnd))
    {
        wstring proc_name = GetProcessNameByHandle(hwnd);
        transform(proc_name.begin(), proc_name.end(), proc_name.begin(), ::tolower);
        return proc_name.find(_T("explorer.exe")) != wstring::npos;
    }
    return false;
}

__declspec(dllexport) void start()
{
    if (!isRunning)
    {
        isRunning = true;
        thread t(_start);
        t.detach();
        thread checkMouse(_checkMouseStatus);
        checkMouse.detach();
    }
}

__declspec(dllexport) void stop()
{
    isRunning = false;
}

__declspec(dllexport) bool is_explorer_at_top()
{
    return isExplorerWindowAtTop;
}

__declspec(dllexport) long getExplorerX()
{
    return explorerX;
}

__declspec(dllexport) long getExplorerY()
{
    return explorerY;
}

__declspec(dllexport) long getExplorerWidth()
{
    return explorerWidth;
}

__declspec(dllexport) long getExplorerHeight()
{
    return explorerHeight;
}

__declspec(dllexport) int get_toolbar_click_x()
{
    return toolbar_click_x;
}

__declspec(dllexport) int get_toolbar_click_y()
{
    return toolbar_click_y;
}

__declspec(dllexport) void resetMouseStatus()
{
    is_click_not_explorer_or_searchbar = false;
}

void _checkMouseStatus()
{
    while (isRunning)
    {
        if (!is_click_not_explorer_or_searchbar) {
            is_click_not_explorer_or_searchbar = isClickNotExplorerOrSearchBarOrSwitchTask();
        }
        Sleep(10);
    }
}

void _start()
{
    HWND hwnd;
    RECT windowRect;
    while (isRunning)
    {
        getTopWindow(&hwnd);
        if (isExplorerWindow(hwnd) || isFileChooserWindow(hwnd))
        {
            getWindowRect(hwnd, &windowRect);
            explorerX = windowRect.left;
            explorerY = windowRect.top;
            explorerWidth = windowRect.right - windowRect.left;
            explorerHeight = windowRect.bottom - windowRect.top;
            if (explorerHeight < EXPLORER_MIN_HEIGHT || explorerWidth < EXPLORER_MIN_WIDTH || explorerX < EXPLORER_MIN_X_POS || explorerY < EXPLORER_MIN_Y_POS) {
                isExplorerWindowAtTop = false;
            }
            else
            {
                setClickPos(hwnd);
                isExplorerWindowAtTop = true;
            }
        }
        else 
        {
            isExplorerWindowAtTop = false;
        }
        if (isExplorerWindowAtTop)
        {
            Sleep(10);
        }
        else
        {
            Sleep(300);
        }
    }
}


#ifdef TEST
int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    start();
    while (true)
    {
        /*if (isExplorerWindowAtTop)
        {
            int xPos = getExplorerX();
            int yPos = getExplorerY();
            int width = getExplorerWidth();
            int height = getExplorerHeight();
            int toolBarX = toolbar_click_x;
            int toolBarY = toolbar_click_y;
            Sleep(20);
        }*/
        if (is_click_not_explorer_or_searchbar)
        {
            Sleep(10);
            break;
        }
        Sleep(10);
    }
}
#endif