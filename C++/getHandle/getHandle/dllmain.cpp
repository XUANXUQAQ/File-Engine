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
//#define TEST


using namespace std;

volatile bool isStart = false;
volatile bool isRunning = true;
int explorerX;
int explorerY;
volatile long explorerWidth;
volatile long explorerHeight;
int toolbar_click_x;
int toolbar_click_y;
volatile bool is_click_not_explorer_or_searchbar;

struct handle_data {
    unsigned long process_id;
    HWND window_handle;
};

void getTopWindow(HWND& hwnd);
void getWindowRect(HWND& hwnd, LPRECT lprect);
bool isExplorerWindow(HWND& hwnd);
void _start();
bool isFileChooserWindow(HWND& hwnd);
void setClickPos(HWND& fileChooserHwnd);
BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam);
bool isClickNotExplorerOrSearchBarOrSwitchTask();
void _checkMouseStatus();
bool isMouseClickedOrSwitchTaskPressed();
bool isSearchBarWindow(HWND& hd);
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

__declspec(dllexport) bool isExplorerAndSearchbarNotFocused()
{
    return is_click_not_explorer_or_searchbar;
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

bool isSearchBarWindow(HWND& hd)
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
    HWND hd = GetDesktopWindow();        //得到桌面窗口
    hd = GetWindow(hd, GW_CHILD);        //得到屏幕上第一个子窗口
    int num = 1;
    while (hd != NULL)                    //循环得到所有的子窗口
    {
        if (IsWindowVisible(hd) && !IsIconic(hd))
        {
            if (isFileChooserWindow(hd) || isExplorerWindow(hd))
            {
                return false;
            }
        }
        hd = GetNextWindow(hd, GW_HWNDNEXT);
    }
    return true;
}

void getTopWindow(HWND& hwnd)
{
    hwnd = ::GetForegroundWindow();
}

void getWindowRect(HWND& hwnd, LPRECT lprect)
{
    GetWindowRect(hwnd, lprect);
}

bool isFileChooserWindow(HWND& hwnd)
{
    char _windowClassName[100];
    char title[100];
    GetClassNameA(hwnd, _windowClassName, 100);
    GetWindowTextA(hwnd, title, 100);
    string windowTitle(title);
    string WindowClassName(_windowClassName);
    transform(windowTitle.begin(), windowTitle.end(), windowTitle.begin(), ::tolower);
    transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
    return (WindowClassName.find("#32770") != string::npos ||
        WindowClassName.find("dialog") != string::npos) && 
        windowTitle.find("internet download manager") == string::npos && 
        windowTitle.find("push commits to") == string::npos &&
        windowTitle.find("geek uninstaller") == string::npos;
}

void setClickPos(HWND& fileChooserHwnd)
{
    EnumChildWindows(fileChooserHwnd, findToolbar, NULL);
}

BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam)
{
    int toolbar_x;
    int toolbar_y;
    int toolbar_width;
    int toolbar_height;
    HWND hwd2 = FindWindowExA(hwndChild, NULL, "Address Band Root", NULL);
    if (hwd2)
    {
        RECT rect;
        getWindowRect(hwd2, &rect);
        toolbar_x = rect.left;
        toolbar_y = rect.top;
        toolbar_width = rect.right - rect.left;
        toolbar_height = rect.bottom - rect.top;
        toolbar_click_x = toolbar_x + toolbar_width - 80;
        toolbar_click_y = toolbar_y + (toolbar_height / 2);
        return false;
    }
    return true;
}

bool isExplorerWindow(HWND& hwnd)
{
    if (IsWindowEnabled(hwnd))
    {
        char className[200];
        GetClassNameA(hwnd, className, 200);
        string WindowClassName(className);
        transform(WindowClassName.begin(), WindowClassName.end(), WindowClassName.begin(), ::tolower);
        return (WindowClassName.find("cabinet") != string::npos ||
            WindowClassName.find("directuihwnd") != string::npos ||
            WindowClassName.find("systreeview") != string::npos ||
            WindowClassName.find("universalsearchband") != string::npos ||
            WindowClassName.find("address band root") != string::npos ||
            WindowClassName.find("netuihwnd") != string::npos ||
            WindowClassName.find("rebarwindow") != string::npos);
    }
    return false;
}

__declspec(dllexport) void start()
{
    thread t(_start);
    t.detach();
    thread checkMouse(_checkMouseStatus);
    checkMouse.detach();
}

__declspec(dllexport) void stop()
{
    isRunning = false;
}

__declspec(dllexport) bool is_explorer_at_top()
{
    return isStart;
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
        getTopWindow(hwnd);
        if (isExplorerWindow(hwnd) || isFileChooserWindow(hwnd))
        {
            getWindowRect(hwnd, &windowRect);
            explorerX = windowRect.left;
            explorerY = windowRect.top;
            explorerWidth = windowRect.right - windowRect.left;
            explorerHeight = windowRect.bottom - windowRect.top;
            if (explorerHeight < 200 || explorerWidth < 200 || explorerX < 50 || explorerY < 50) {
                isStart = false;
            }
            else
            {
                setClickPos(hwnd);
                isStart = true;
            }
        }
        else 
        {
            isStart = false;
        }
        if (isStart)
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
        /*if (isStart)
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