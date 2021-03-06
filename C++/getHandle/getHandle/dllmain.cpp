﻿// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <Windows.h>
#include <iostream>
#include <TlHelp32.h>
#include <tchar.h>
#include <thread>
#include <dwmapi.h>
#include "checkHwnd.h"
#include "getExplorerPath.h"
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "dwmapi")
#pragma comment(lib, "user32")
#pragma comment(lib, "kernel32")
//#define TEST

using namespace std;

constexpr auto EXPLORER_MIN_HEIGHT = 200;       //当窗口大小满足这些条件后才开始判断是否为explorer.exe
constexpr auto EXPLORER_MIN_WIDTH = 200;

constexpr auto G_DIALOG = 1;
constexpr auto G_EXPLORER = 2;

volatile bool isExplorerWindowAtTop = false;
volatile bool isMouseClickOutOfExplorer = false;
volatile bool isRunning = false;
volatile int explorerX;
volatile int explorerY;
volatile long explorerWidth;
volatile long explorerHeight;
volatile int toolbar_click_x;
volatile int toolbar_click_y;
volatile int topWindowType;
HWND currentAttachExplorer;
char dragExplorerPath[500];

void checkTopWindowThread();
void checkMouseThread();
inline void setClickPos(const HWND& fileChooserHwnd);
BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam);
inline bool isMouseClicked();
inline bool isKeyPressed(int vk_key);
bool isDialogNotExist();
extern "C" __declspec(dllexport) void start();
extern "C" __declspec(dllexport) void stop();
extern "C" __declspec(dllexport) bool changeToAttach();
extern "C" __declspec(dllexport) bool changeToNormal();
extern "C" __declspec(dllexport) long getExplorerX();
extern "C" __declspec(dllexport) long getExplorerY();
extern "C" __declspec(dllexport) long getExplorerWidth();
extern "C" __declspec(dllexport) long getExplorerHeight();
extern "C" __declspec(dllexport) const char* getExplorerPath();
extern "C" __declspec(dllexport) bool isDialogWindow();
extern "C" __declspec(dllexport) void bringSearchBarToTop();
extern "C" __declspec(dllexport) int getToolBarX();
extern "C" __declspec(dllexport) int getToolBarY();
extern "C" __declspec(dllexport) double getDpi();
extern "C" __declspec(dllexport) bool isMousePressed();

#ifdef TEST
void outputHwndInfo(HWND hwnd)
{
    char hwndTitle[200];
    char hwndClassName[200];
    GetWindowTextA(hwnd, hwndTitle, 200);
    GetClassNameA(hwnd, hwndClassName, 200);
    cout << "hwnd class name: " << hwndClassName << endl;
    cout << "hwnd title: " << hwndTitle << endl;
}
#endif

int getToolBarX()
{
    return toolbar_click_x;
}

bool isMousePressed()
{
	if (isKeyPressed(VK_LBUTTON) || isKeyPressed(VK_RBUTTON))
	{
        return true;
	}
    return false;
}

double getDpi()
{
    SetProcessDPIAware();
    // Get desktop dc
    HDC desktopDc = GetDC(nullptr);
    // Get native resolution
    const int dpi = GetDeviceCaps(desktopDc, LOGPIXELSX);
    auto ret = 1 + static_cast<double>((dpi - 96) / 24) * 0.25;
    if (ret < 1)
    {
        ret = 1;
    }

    ReleaseDC(nullptr, desktopDc);
    return ret;
}

int getToolBarY()
{
    return toolbar_click_y;
}

bool changeToNormal()
{
    return isMouseClickOutOfExplorer || isDialogNotExist();
}

void bringSearchBarToTop()
{
    auto* const hWnd = getSearchBarHWND();
    auto* hForeWnd = GetForegroundWindow();
    const auto dwForeID = GetWindowThreadProcessId(hForeWnd, nullptr);
    const auto dwCurID = GetCurrentThreadId();
    AttachThreadInput(dwCurID, dwForeID, TRUE);
    ShowWindow(hWnd, SW_SHOWNORMAL);
    SetWindowPos(hWnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
    SetWindowPos(hWnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
    SetForegroundWindow(hWnd);
    AttachThreadInput(dwCurID, dwForeID, FALSE);
}

bool isDialogWindow()
{
    return topWindowType == G_DIALOG;
}

const char* getExplorerPath()
{
    POINT p;
    GetCursorPos(&p);
    auto* hd = WindowFromPoint(p);
    hd = GetAncestor(hd, GA_ROOT);
    strcpy_s(dragExplorerPath, getPathByHWND(hd).c_str());
    return dragExplorerPath;
}

inline bool isKeyPressed(const int vk_key)
{
    return GetAsyncKeyState(vk_key) & 0x8000 ? true : false;
}

inline bool isMouseClicked()
{
    return isKeyPressed(VK_RBUTTON) || 
        isKeyPressed(VK_MBUTTON) || 
        isKeyPressed(VK_LBUTTON);
}

bool isDialogNotExist()
{
    RECT windowRect;
    auto* hd = GetDesktopWindow();        //得到桌面窗口
    hd = GetWindow(hd, GW_CHILD);        //得到屏幕上第一个子窗口
    while (hd != nullptr)                    //循环得到所有的子窗口
    {
        if (IsWindowVisible(hd) && !IsIconic(hd))
        {
            GetWindowRect(hd, &windowRect);
            const int tmp_explorerWidth = windowRect.right - windowRect.left;
            const int tmp_explorerHeight = windowRect.bottom - windowRect.top;
            if (!(tmp_explorerHeight < EXPLORER_MIN_HEIGHT || tmp_explorerWidth < EXPLORER_MIN_WIDTH))
            {
                if (is_explorer_window_low_cost(hd) || is_file_chooser_window(hd))
                {
                    return false;
                }
            }
        }
        hd = GetWindow(hd, GW_HWNDNEXT);
    }
#ifdef TEST
    cout << "all dialog not exist" << endl;
#endif
    return true;
}

inline void setClickPos(const HWND& fileChooserHwnd)
{
    EnumChildWindows(fileChooserHwnd, findToolbar, NULL);
}

BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam)
{
	auto* hwd2 = FindWindowExA(hwndChild, nullptr, "Address Band Root", nullptr);
    if (IsWindow(hwd2))
    {
        RECT rect;
        GetWindowRect(hwd2, &rect);
        const int toolbar_x = rect.left;
        const int toolbar_y = rect.top;
        const int toolbar_width = rect.right - rect.left;
        const int toolbar_height = rect.bottom - rect.top;
        toolbar_click_x = toolbar_x + toolbar_width - 100;
        toolbar_click_y = toolbar_y + toolbar_height / 2;
        return false;
    }
    return true;
}

void start()
{
    if (!isRunning)
    {
        isRunning = true;
        thread checkTopWindow(checkTopWindowThread);
        checkTopWindow.detach();
        thread checkMouse(checkMouseThread);
        checkMouse.detach();
    }
}

void stop()
{
    isRunning = false;
}

bool changeToAttach()
{
    return isExplorerWindowAtTop;
}

long getExplorerX()
{
    return explorerX;
}

long getExplorerY()
{
    return explorerY;
}

long getExplorerWidth()
{
    return explorerWidth;
}

long getExplorerHeight()
{
    return explorerHeight;
}

void checkMouseThread()
{
    POINT point;
    RECT explorerArea;
    RECT searchBarArea;
    int count = 0;
    constexpr int waitCountTimes = 25;
    constexpr int maxWaitCount = waitCountTimes * 2;
    bool isMouseClickedFlag = false;
	while (isRunning)
	{
		if (count <= maxWaitCount)
		{
            count++;
		}
		if (isMouseClickedFlag && count > waitCountTimes && !isMouseClickOutOfExplorer || count > maxWaitCount)
		{
            count = 0;
            isMouseClickedFlag = false;
            HWND topWindow = GetForegroundWindow();
            isMouseClickOutOfExplorer = !(is_explorer_window_high_cost(topWindow) || is_file_chooser_window(topWindow) || is_search_bar_window(topWindow));
		}
		if (!IsWindow(currentAttachExplorer) || IsIconic(currentAttachExplorer))
		{
            isMouseClickOutOfExplorer = true;
		} else if (isMouseClicked()) {
            if (GetCursorPos(&point))
            {
                GetWindowRect(currentAttachExplorer, &explorerArea);
                GetWindowRect(getSearchBarHWND(), &searchBarArea);
                isMouseClickOutOfExplorer = 
                    !(explorerArea.left <= point.x && point.x <= explorerArea.right && (explorerArea.top <= point.y && point.y <= explorerArea.bottom)) &&
                    !(searchBarArea.left <= point.x && point.x <= searchBarArea.right && (searchBarArea.top <= point.y && point.y <= searchBarArea.bottom));
#ifdef TEST
                cout << "point X:" << point.x << endl;
                cout << "point Y:" << point.y << endl;
                cout << "left :" << explorerArea.left << "  right :" << explorerArea.right << "  top :" << explorerArea.top << "  bottom :" << explorerArea.bottom << endl;
                if (explorerArea.left <= point.x && point.x <= explorerArea.right && (explorerArea.top <= point.y && point.y <= explorerArea.bottom))
                {
                    cout << "return false" << endl;
                }
#endif
            }
            count = 0;
            isMouseClickedFlag = true;
        }
		if (IsWindowVisible(getSearchBarHWND())) {
            Sleep(10);
		} else {
#ifdef TEST
            cout << "search bar not visible" << endl;
#endif
            Sleep(300);
		}
	}
}

void checkTopWindowThread()
{
	RECT windowRect;
    while (isRunning)
    {
        HWND hwnd = GetForegroundWindow();
        const auto isExplorerWindow = is_explorer_window_low_cost(hwnd);
        const auto isDialogWindow = is_file_chooser_window(hwnd);

        if (isExplorerWindow || isDialogWindow)
        {
            GetWindowRect(hwnd, &windowRect);
            if (IsZoomed(hwnd)) {
                explorerX = 0;
                explorerY = 0;
            } else {
                explorerX = windowRect.left;
                explorerY = windowRect.top;
            }
            explorerWidth = windowRect.right - windowRect.left;
            explorerHeight = windowRect.bottom - windowRect.top;
            if (explorerHeight < EXPLORER_MIN_HEIGHT || explorerWidth < EXPLORER_MIN_WIDTH) {
                isExplorerWindowAtTop = false;
            }
            else
            {
                if (isExplorerWindow)
                {
                    topWindowType = G_EXPLORER;
                }
                else if (isDialogWindow)
                {
                    topWindowType = G_DIALOG;
                }
                currentAttachExplorer = hwnd;
                setClickPos(hwnd);
                isExplorerWindowAtTop = true;
                isMouseClickOutOfExplorer = false;
            }
        }
        else
        {
            isExplorerWindowAtTop = false;
        }
        if (isExplorerWindowAtTop)
        {
            Sleep(5);
        }
        else
        {
            Sleep(300);
        }
    }
}

