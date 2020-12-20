// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <Windows.h>
#include <iostream>
#include <TlHelp32.h>
#include <tchar.h>
#include <thread>
#include <dwmapi.h>
#include "checkHwnd.h"
#include "getExplorerPath.h"
#pragma comment(lib, "dwmapi")
#pragma comment(lib, "user32")
#pragma comment(lib, "kernel32")
#define IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0)
//#define TEST


constexpr auto EXPLORER_MIN_HEIGHT = 200;       //当窗口大小满足这些条件后才开始判断是否为explorer.exe
constexpr auto EXPLORER_MIN_WIDTH = 200;
constexpr auto EXPLORER_MIN_X_POS = 50;
constexpr auto EXPLORER_MIN_Y_POS = 50;

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
char explorer_path[1000];
volatile bool isSearchBarUsing = false;

void getTopWindow(HWND* hwnd);
void getWindowRect(const HWND& hwnd, LPRECT lprect);
void checkTopWindowThread();
void setClickPos(const HWND& fileChooserHwnd);
BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam);
BOOL CALLBACK findSeachBar(HWND hwndChild, LPARAM lParam);
bool isClickNotExplorerOrSearchBarOrSwitchTask();
void checkMouseClickHWND();
bool isMouseClickedOrSwitchTaskPressed();
void grabSearchBarHWND();

extern "C" __declspec(dllexport) bool isDialogNotExist();
extern "C" __declspec(dllexport) void start();
extern "C" __declspec(dllexport) void stop();
extern "C" __declspec(dllexport) bool isExplorerAtTop();
extern "C" __declspec(dllexport) long getExplorerX();
extern "C" __declspec(dllexport) long getExplorerY();
extern "C" __declspec(dllexport) long getExplorerWidth();
extern "C" __declspec(dllexport) long getExplorerHeight();
extern "C" __declspec(dllexport) int getToolbarClickX();
extern "C" __declspec(dllexport) int getToolbarClickY(); 
extern "C" __declspec(dllexport) bool isExplorerAndSearchbarNotFocused();
extern "C" __declspec(dllexport) void setExplorerPath();
extern "C" __declspec(dllexport) const char* getExplorerPath();
extern "C" __declspec(dllexport) void setSearchBarUsingStatus(bool b);

__declspec(dllexport) void setSearchBarUsingStatus(bool b)
{
    isSearchBarUsing = b;
}

__declspec(dllexport) void setExplorerPath()
{
    POINT p;
    GetCursorPos(&p);
    HWND hd = ::WindowFromPoint(p);
    hd = GetAncestor(hd, GA_ROOT);
    string path_tmp = getPathByHWND(hd);
#ifdef TEST
    char windowClassName[200];
    GetClassNameA(hd, windowClassName, 200);
    std::cout << "class name:" << windowClassName << endl;
#endif
    strcpy_s(explorer_path, 1000, path_tmp.c_str());
}

__declspec(dllexport) const char* getExplorerPath()
{
    return explorer_path;
}

__declspec(dllexport) bool isExplorerAndSearchbarNotFocused()
{
    return is_click_not_explorer_or_searchbar;
}

void grabSearchBarHWND()
{
    EnumWindows(findSeachBar, NULL);
}

bool isClickNotExplorerOrSearchBarOrSwitchTask()
{
    POINT point;
    HWND hd;    //鼠标位置的窗口句柄
    if (isMouseClickedOrSwitchTaskPressed())
    {
        if (GetCursorPos(&point))
        {
            hd = WindowFromPoint(point);
            if (isExplorerWindowHighCost(hd) || is_file_chooser_window(hd))
            {
                return false;
            } 
	           //检查是否点击搜索框
	          return !(is_search_bar_window(hd));
        }
    }
    else
    {
        getTopWindow(&hd);
        return !(is_file_chooser_window(hd) || is_explorer_window_low_cost(hd) || is_search_bar_window(hd));
    }
    return false;
}

bool isMouseClickedOrSwitchTaskPressed()
{
    return IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_RBUTTON) || IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_MBUTTON) || IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_LBUTTON) ||
        (IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_MENU) && IS_MOUSE_CLICKED_OR_KEY_PRESSED(VK_TAB));
}

__declspec(dllexport) bool isDialogNotExist()
{
    RECT windowRect;
    HWND hd = GetDesktopWindow();        //得到桌面窗口
    hd = GetWindow(hd, GW_CHILD);        //得到屏幕上第一个子窗口
    while (hd != nullptr)                    //循环得到所有的子窗口
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
                if (is_explorer_window_low_cost(hd) || is_file_chooser_window(hd))
                {
                    return false;
                }
            }
        }
        hd = GetWindow(hd, GW_HWNDNEXT);
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

void setClickPos(const HWND& fileChooserHwnd)
{
    EnumChildWindows(fileChooserHwnd, findToolbar, NULL);
}

BOOL CALLBACK findSeachBar(HWND hwndChild, LPARAM lParam)
{
    if (is_search_bar_window(hwndChild))
    {
        searchBarHWND = hwndChild;
        return false;
    }
    return true;
}

BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam)
{
    HWND hwd2 = FindWindowExA(hwndChild, nullptr, "Address Band Root", nullptr);
    if (IsWindow(hwd2))
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


__declspec(dllexport) void start()
{
    if (!isRunning)
    {
        isRunning = true;
        thread t(checkTopWindowThread);
        t.detach();
        thread checkMouse(checkMouseClickHWND);
        checkMouse.detach();
    }
}

__declspec(dllexport) void stop()
{
    isRunning = false;
}

__declspec(dllexport) bool isExplorerAtTop()
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

__declspec(dllexport) int getToolbarClickX()
{
    return toolbar_click_x;
}

__declspec(dllexport) int getToolbarClickY()
{
    return toolbar_click_y;
}

void checkMouseClickHWND()
{
    while (isRunning)
    {
    	if (isSearchBarUsing)
    	{
            is_click_not_explorer_or_searchbar = isClickNotExplorerOrSearchBarOrSwitchTask();
    	}
        Sleep(10);
    }
}

void checkTopWindowThread()
{
    HWND hwnd;
    RECT windowRect;
    while (isRunning)
    {
        getTopWindow(&hwnd);
        if (is_explorer_window_low_cost(hwnd) || is_file_chooser_window(hwnd))
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

