// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <Windows.h>
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

using namespace std;

constexpr auto EXPLORER_MIN_HEIGHT = 200; //当窗口大小满足这些条件后才开始判断是否为explorer.exe
constexpr auto EXPLORER_MIN_WIDTH = 200;

constexpr auto G_DIALOG = 1;
constexpr auto G_EXPLORER = 2;

volatile bool is_explorer_window_at_top = false;
volatile bool is_mouse_click_out_of_explorer = false;
volatile bool is_running = false;
volatile int explorer_x;
volatile int explorer_y;
volatile long explorer_width;
volatile long explorer_height;
volatile int toolbar_click_x;
volatile int toolbar_click_y;
volatile int toolbar_width;
volatile int toolbar_height;
volatile int top_window_type;
HWND current_attach_explorer;
char drag_explorer_path[500];

void checkTopWindowThread();
void checkMouseThread();
inline void setClickPos(const HWND& fileChooserHwnd);
BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam);
inline bool isMouseClicked();
bool isDialogNotExist();
BOOL CALLBACK findToolbarWin32Internal(HWND hwndChild, LPARAM lParam);

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
__declspec(dllexport) void setEditPath(const jchar* path);
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

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    setEditPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_setEditPath
(JNIEnv* env, jobject, jstring path)
{
	const jchar* _tmp_path = env->GetStringChars(path, nullptr);
	setEditPath(_tmp_path);
	env->ReleaseStringChars(path, _tmp_path);
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
	return is_mouse_click_out_of_explorer || isDialogNotExist();
}

BOOL isDialogWindow()
{
	return top_window_type == G_DIALOG;
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
	strcpy_s(drag_explorer_path, getPathByHWND(hd).c_str());
	return drag_explorer_path;
}

/**
 * 检查键盘是否按下
 */
BOOL isKeyPressed(const int vk_key)
{
	return GetAsyncKeyState(vk_key) & 0x8000 ? TRUE : FALSE;
}

/**
 * 检测鼠标的点击
 */
inline bool isMouseClicked()
{
	return isKeyPressed(VK_RBUTTON) ||
		isKeyPressed(VK_MBUTTON) ||
		isKeyPressed(VK_LBUTTON);
}

/**
 * 判断窗口句柄是否已经失效，即被关闭
 */
bool isDialogNotExist()
{
	RECT window_rect;
	auto* hd = GetDesktopWindow(); //得到桌面窗口
	hd = GetWindow(hd, GW_CHILD); //得到屏幕上第一个子窗口
	while (hd != nullptr) //循环得到所有的子窗口
	{
		if (IsWindowVisible(hd) && !IsIconic(hd))
		{
			GetWindowRect(hd, &window_rect);
			const int tmp_explorerWidth = window_rect.right - window_rect.left;
			const int tmp_explorerHeight = window_rect.bottom - window_rect.top;
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

/**
 * 定位explorer窗口输入文件夹路径的控件坐标
 */
inline void setClickPos(const HWND& fileChooserHwnd)
{
	EnumChildWindows(fileChooserHwnd, findToolbar, NULL);
}

void setEditPath(const jchar* path)
{
	using namespace std::chrono;
	const POINT toolbar_pos{toolbar_click_x, toolbar_click_y};
	POINT origin_mouse_pos;
	GetCursorPos(&origin_mouse_pos);
	char class_name[50] = {"\0"};
	const auto start_time = system_clock::now();
	const auto end_time = start_time + seconds(3);
	HWND hwnd_from_toolbar;
	bool is_timeout = false;
	const POINT toolbar_center{ toolbar_click_x - toolbar_width / 2, toolbar_click_y };
	while (true)
	{
		//尝试点击
		SetCursorPos(toolbar_pos.x, toolbar_pos.y);
		mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
		Sleep(15);
		SetCursorPos(origin_mouse_pos.x, origin_mouse_pos.y);

		//检查toolbar位置是否已经变成Edit框
		hwnd_from_toolbar = WindowFromPoint(toolbar_center);
		GetClassNameA(hwnd_from_toolbar, class_name, 50);
		if (strcmp(class_name, "Edit") == 0)
		{
			break;
		}
		
		if (system_clock::now() > end_time)
		{
			fprintf(stderr, "Get explorer edit path timeout.");
			is_timeout = true;
			break;
		}
	}
	if (is_timeout)
	{
		return;
	}
	SendMessageW(hwnd_from_toolbar, EM_SETSEL, static_cast<WPARAM>(0), -1);
	SendMessageW(hwnd_from_toolbar, EM_REPLACESEL, static_cast<WPARAM>(TRUE), reinterpret_cast<LPARAM>(path));
	INPUT input{};
	input.type = INPUT_KEYBOARD;
	input.ki.wVk = VK_RETURN;
	SendInput(1, &input, sizeof(INPUT));
}

BOOL CALLBACK findToolbarWin32Internal(HWND hwndChild, LPARAM lParam)
{
	auto* hwd_tool_bar = FindWindowExA(hwndChild, nullptr, "ToolbarWindow32", nullptr);
	if (IsWindow(hwd_tool_bar))
	{
		RECT combo_box_rect;
		GetWindowRect(hwd_tool_bar, &combo_box_rect);
		*reinterpret_cast<int*>(lParam) = combo_box_rect.right - combo_box_rect.left;
		return false;
	}
	return true;
}

/**
 * 定位explorer窗口输入文件夹路径的控件坐标
 */
BOOL CALLBACK findToolbar(HWND hwndChild, LPARAM lParam)
{
	auto* hwd2 = FindWindowExA(hwndChild, nullptr, "Address Band Root", nullptr);
	if (IsWindow(hwd2))
	{
		RECT rect;
		int toolbar_win32_width = 0;
		GetWindowRect(hwd2, &rect);
		const int toolbar_x = rect.left;
		const int toolbar_y = rect.top;
		toolbar_width = rect.right - rect.left;
		toolbar_height = rect.bottom - rect.top;
		EnumChildWindows(hwd2, findToolbarWin32Internal, reinterpret_cast<LPARAM>(&toolbar_win32_width));
		const int combo_box_width = toolbar_win32_width;
		toolbar_click_x = toolbar_x + toolbar_width - combo_box_width - 15;
		toolbar_click_y = toolbar_y + toolbar_height / 2;
		return false;
	}
	return true;
}

/**
 * 开始检测窗口句柄
 */
void start()
{
	if (!is_running)
	{
		is_running = true;
		thread checkTopWindow(checkTopWindowThread);
		checkTopWindow.detach();
		thread checkMouse(checkMouseThread);
		checkMouse.detach();
	}
}

/**
 * 停止检测
 */
void stop()
{
	is_running = false;
}

/**
 * 判断File-Engine窗口是否切换到贴靠模式
 */
BOOL changeToAttach()
{
	return is_explorer_window_at_top;
}

/**
 * 获取explorer窗口左上角的X坐标
 */
long getExplorerX()
{
	return explorer_x;
}

/**
 * 获取explorer窗口左上角的Y坐标
 */
long getExplorerY()
{
	return explorer_y;
}

/**
 * 获得explorer窗口的宽度
 */
long getExplorerWidth()
{
	return explorer_width;
}

/**
 * 获得explorer窗口的高度
 */
long getExplorerHeight()
{
	return explorer_height;
}

void checkMouseThread()
{
	POINT point;
	RECT explorer_area;
	RECT search_bar_area;
	int count = 0;
	constexpr int wait_count_times = 25;
	constexpr int max_wait_count = wait_count_times * 2;
	bool is_mouse_clicked_flag = false;
	while (is_running)
	{
		if (count <= max_wait_count)
		{
			count++;
		}
		// count防止窗口闪烁
		if (is_mouse_clicked_flag && count > wait_count_times && !is_mouse_click_out_of_explorer || count >
			max_wait_count)
		{
			count = 0;
			is_mouse_clicked_flag = false;
			HWND top_window = GetForegroundWindow();
			is_mouse_click_out_of_explorer = !(is_explorer_window_by_process(top_window) || is_file_chooser_window(
					top_window)
				|| is_search_bar_window(top_window));
		}
		// 如果窗口句柄已经失效或者最小化，则判定为关闭窗口
		if (!IsWindow(current_attach_explorer) || IsIconic(current_attach_explorer))
		{
			is_mouse_click_out_of_explorer = true;
		}
		else if (isMouseClicked())
		{
			// 检测鼠标位置，如果点击位置不在explorer窗口内则判定为关闭窗口
			if (GetCursorPos(&point))
			{
				GetWindowRect(current_attach_explorer, &explorer_area);
				GetWindowRect(get_search_bar_hwnd(), &search_bar_area);
				is_mouse_click_out_of_explorer =
					!(explorer_area.left <= point.x && point.x <= explorer_area.right && (explorer_area.top <= point.y
						&&
						point.y <= explorer_area.bottom)) &&
					!(search_bar_area.left <= point.x && point.x <= search_bar_area.right && (search_bar_area.top <=
						point.y
						&& point.y <= search_bar_area.bottom));
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
			is_mouse_clicked_flag = true;
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
	RECT window_rect;
	while (is_running)
	{
		HWND hwnd = GetForegroundWindow();
		const auto isExplorerWindow = is_explorer_window_by_class_name(hwnd);
		const auto isDialogWindow = is_file_chooser_window(hwnd);

		if (isExplorerWindow || isDialogWindow)
		{
			GetWindowRect(hwnd, &window_rect);
			if (IsZoomed(hwnd))
			{
				explorer_x = 0;
				explorer_y = 0;
			}
			else
			{
				explorer_x = window_rect.left;
				explorer_y = window_rect.top;
			}
			explorer_width = window_rect.right - window_rect.left;
			explorer_height = window_rect.bottom - window_rect.top;
			if (explorer_height < EXPLORER_MIN_HEIGHT || explorer_width < EXPLORER_MIN_WIDTH)
			{
				is_explorer_window_at_top = false;
			}
			else
			{
				if (isExplorerWindow)
				{
					top_window_type = G_EXPLORER;
				}
				else if (isDialogWindow)
				{
					top_window_type = G_DIALOG;
				}
				current_attach_explorer = hwnd;
				setClickPos(hwnd);
				is_explorer_window_at_top = true;
				is_mouse_click_out_of_explorer = false;
			}
		}
		else
		{
			is_explorer_window_at_top = false;
		}
		if (is_explorer_window_at_top)
		{
			Sleep(5);
		}
		else
		{
			Sleep(300);
		}
	}
}
