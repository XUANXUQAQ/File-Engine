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
#include "file_engine_dllInterface_GetHandle.h"
#pragma comment(lib, "dwmapi")
#pragma comment(lib, "user32")
#pragma comment(lib, "kernel32")
//#define TEST

using namespace std;

constexpr auto EXPLORER_MIN_HEIGHT = 200; //当窗口大小满足这些条件后才开始判断是否为explorer.exe
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
bool isDialogNotExist();

extern "C" {
__declspec(dllexport) void start();
__declspec(dllexport) void stop();
__declspec(dllexport) BOOL changeToAttach();
__declspec(dllexport) BOOL changeToNormal();
__declspec(dllexport) long getExplorerX();
__declspec(dllexport) long getExplorerY();
__declspec(dllexport) long getExplorerWidth();
__declspec(dllexport) long getExplorerHeight();
__declspec(dllexport) const char* getExplorerPath();
__declspec(dllexport) BOOL isDialogWindow();
__declspec(dllexport) int getToolBarX();
__declspec(dllexport) int getToolBarY();
__declspec(dllexport) BOOL isKeyPressed(int vk_key);
__declspec(dllexport) BOOL isForegroundFullscreen();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    start
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_start
(JNIEnv*, jobject)
{
	start();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    stop
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_stop
(JNIEnv*, jobject)
{
	stop();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    changeToAttach
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_changeToAttach
(JNIEnv*, jobject)
{
	return static_cast<jboolean>(changeToAttach());
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    changeToNormal
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_changeToNormal
(JNIEnv*, jobject)
{
	return static_cast<jboolean>(changeToNormal());
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerX
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerX
(JNIEnv*, jobject)
{
	return getExplorerX();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerY
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerY
(JNIEnv*, jobject)
{
	return getExplorerY();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerWidth
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerWidth
(JNIEnv*, jobject)
{
	return getExplorerWidth();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerHeight
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerHeight
(JNIEnv*, jobject)
{
	return getExplorerHeight();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerPath
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerPath
(JNIEnv* env, jobject)
{
	const char* tmp = getExplorerPath();
	return env->NewStringUTF(tmp);
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    isDialogWindow
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_isDialogWindow
(JNIEnv*, jobject)
{
	return static_cast<jboolean>(isDialogWindow());
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getToolBarX
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetHandle_getToolBarX
(JNIEnv*, jobject)
{
	return getToolBarX();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getToolBarY
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetHandle_getToolBarY
(JNIEnv*, jobject)
{
	return getToolBarY();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    isKeyPressed
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_isKeyPressed
(JNIEnv*, jobject, jint vk_key)
{
	return static_cast<jboolean>(isKeyPressed(vk_key));
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    isForegroundFullscreen
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_isForegroundFullscreen
(JNIEnv*, jobject)
{
	return static_cast<jboolean>(isForegroundFullscreen());
}


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

/**
 * 检查顶层窗口是不是全屏显示，防止游戏时开始搜索导致卡顿
 */
BOOL isForegroundFullscreen()
{
	bool b_fullscreen = false; //存放当前激活窗口是否是全屏的，true表示是，false表示不是
	RECT rc_app;
	RECT rc_desk;

	HWND hWnd = GetForegroundWindow(); //获取当前正在与用户交互的当前激活窗口句柄

	if ((hWnd != GetDesktopWindow()) && (hWnd != GetShellWindow())) //如果当前激活窗口不是桌面窗口，也不是控制台窗口
	{
		GetWindowRect(hWnd, &rc_app); //获取当前激活窗口的坐标
		GetWindowRect(GetDesktopWindow(), &rc_desk); //根据桌面窗口句柄，获取整个屏幕的坐标

		if (rc_app.left <= rc_desk.left && //如果当前激活窗口的坐标完全覆盖住桌面窗口，就表示当前激活窗口是全屏的
			rc_app.top <= rc_desk.top &&
			rc_app.right >= rc_desk.right &&
			rc_app.bottom >= rc_desk.bottom)
		{
			char szTemp[100];

			if (GetClassNameA(hWnd, szTemp, sizeof(szTemp)) > 0) //如果获取当前激活窗口的类名成功
			{
				if (strcmp(szTemp, "WorkerW") != 0) //如果不是桌面窗口的类名，就认为当前激活窗口是全屏窗口
					b_fullscreen = true;
			}
			else
			{
				b_fullscreen = true; //如果获取失败，就认为当前激活窗口是全屏窗口
			}
		}
	} //如果当前激活窗口是桌面窗口，或者是控制台窗口，就直接返回不是全屏

	return b_fullscreen;
}

int getToolBarY()
{
	return toolbar_click_y;
}

BOOL changeToNormal()
{
	return isMouseClickOutOfExplorer || isDialogNotExist();
}

BOOL isDialogWindow()
{
	return topWindowType == G_DIALOG;
}

/**
 * 获取鼠标当前位置的explorer窗口句柄打开的文件夹位置
 */
const char* getExplorerPath()
{
	POINT p;
	GetCursorPos(&p);
	auto* hd = WindowFromPoint(p);
	hd = GetAncestor(hd, GA_ROOT);
	strcpy_s(dragExplorerPath, getPathByHWND(hd).c_str());
	return dragExplorerPath;
}

BOOL isKeyPressed(const int vk_key)
{
	return GetAsyncKeyState(vk_key) & 0x8000 ? TRUE : FALSE;
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
	auto* hd = GetDesktopWindow(); //得到桌面窗口
	hd = GetWindow(hd, GW_CHILD); //得到屏幕上第一个子窗口
	while (hd != nullptr) //循环得到所有的子窗口
	{
		if (IsWindowVisible(hd) && !IsIconic(hd))
		{
			GetWindowRect(hd, &windowRect);
			const int tmp_explorerWidth = windowRect.right - windowRect.left;
			const int tmp_explorerHeight = windowRect.bottom - windowRect.top;
			if (!(tmp_explorerHeight < EXPLORER_MIN_HEIGHT || tmp_explorerWidth < EXPLORER_MIN_WIDTH))
			{
				if (is_explorer_window_by_class_name(hd) || is_file_chooser_window(hd))
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

BOOL changeToAttach()
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
		// count防止窗口闪烁
		if (isMouseClickedFlag && count > waitCountTimes && !isMouseClickOutOfExplorer || count > maxWaitCount)
		{
			count = 0;
			isMouseClickedFlag = false;
			HWND topWindow = GetForegroundWindow();
			isMouseClickOutOfExplorer = !(is_explorer_window_by_process(topWindow) || is_file_chooser_window(topWindow)
				|| is_search_bar_window(topWindow));
		}
		// 如果窗口句柄已经失效或者最小化，则判定为关闭窗口
		if (!IsWindow(currentAttachExplorer) || IsIconic(currentAttachExplorer))
		{
			isMouseClickOutOfExplorer = true;
		}
		else if (isMouseClicked())
		{
			// 检测鼠标位置，如果点击位置不在explorer窗口内则判定为关闭窗口
			if (GetCursorPos(&point))
			{
				GetWindowRect(currentAttachExplorer, &explorerArea);
				GetWindowRect(get_search_bar_hwnd(), &searchBarArea);
				isMouseClickOutOfExplorer =
					!(explorerArea.left <= point.x && point.x <= explorerArea.right && (explorerArea.top <= point.y &&
						point.y <= explorerArea.bottom)) &&
					!(searchBarArea.left <= point.x && point.x <= searchBarArea.right && (searchBarArea.top <= point.y
						&& point.y <= searchBarArea.bottom));
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
		if (IsWindowVisible(get_search_bar_hwnd()))
		{
			Sleep(10);
		}
		else
		{
#ifdef TEST
            cout << "search bar not visible" << endl;
#endif
			Sleep(300);
		}
	}
}

/**
 * 检查顶层窗口类型，是选择文件对话框还是explorer
 */
void checkTopWindowThread()
{
	RECT windowRect;
	while (isRunning)
	{
		HWND hwnd = GetForegroundWindow();
		const auto isExplorerWindow = is_explorer_window_by_class_name(hwnd);
		const auto isDialogWindow = is_file_chooser_window(hwnd);

		if (isExplorerWindow || isDialogWindow)
		{
			GetWindowRect(hwnd, &windowRect);
			if (IsZoomed(hwnd))
			{
				explorerX = 0;
				explorerY = 0;
			}
			else
			{
				explorerX = windowRect.left;
				explorerY = windowRect.top;
			}
			explorerWidth = windowRect.right - windowRect.left;
			explorerHeight = windowRect.bottom - windowRect.top;
			if (explorerHeight < EXPLORER_MIN_HEIGHT || explorerWidth < EXPLORER_MIN_WIDTH)
			{
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
