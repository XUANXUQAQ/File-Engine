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
#include "quick_jump.h"
#pragma comment(lib, "dwmapi")
#pragma comment(lib, "user32")
#pragma comment(lib, "kernel32")


constexpr auto EXPLORER_MIN_HEIGHT = 200; //当窗口大小满足这些条件后才开始判断是否为explorer.exe
constexpr auto EXPLORER_MIN_WIDTH = 200;

constexpr auto G_DIALOG = 1;
constexpr auto G_EXPLORER = 2;

bool is_explorer_window_at_top = false;
bool is_mouse_click_out_of_explorer = false;
bool is_running = false;
int explorer_x;
int explorer_y;
long explorer_width;
long explorer_height;
int top_window_type;
HWND current_attach_explorer;
char drag_explorer_path[500];

void checkTopWindowThread();
void checkMouseThread();
inline bool isMouseClicked();
bool isDialogNotExist();
void start();
void stop();
BOOL changeToAttach();
BOOL changeToNormal();
long getExplorerX();
long getExplorerY();
long getExplorerWidth();
long getExplorerHeight();
const char* getExplorerPath();
BOOL isDialogWindow();
BOOL isKeyPressed(int vk_key);
BOOL isForegroundFullscreen();


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
    return 0;
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getToolBarY
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetHandle_getToolBarY
(JNIEnv*, jobject)
{
    return 0;
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
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_setEditPath
(JNIEnv* env, jobject, jstring path, jstring file_name)
{
    std::wstring path_wstring;
    std::wstring file_name_wstring;

    const jchar* raw = env->GetStringChars(path, nullptr);
    const jchar* raw_name = env->GetStringChars(file_name, nullptr);

    jsize len = env->GetStringLength(path);
    jsize len_name = env->GetStringLength(file_name);

    path_wstring.assign(raw, raw + len);
    file_name_wstring.assign(raw_name, raw_name + len_name);

    env->ReleaseStringChars(path, raw);
    env->ReleaseStringChars(file_name, raw_name);

    CoInitialize(nullptr);
    jump_to_dest(current_attach_explorer, path_wstring.c_str());
    set_file_selected(current_attach_explorer, file_name_wstring.c_str());
    CoUninitialize();
}

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    bringWindowToTop
 * Signature: ()V
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_bringWindowToTop
(JNIEnv*, jobject)
{
    const auto search_bar_hwnd = get_search_bar_hwnd();
    const DWORD dwCurID = GetCurrentThreadId();
    const DWORD dwForeID = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
    AttachThreadInput(dwCurID, dwForeID, TRUE);
    Sleep(20);
    SetForegroundWindow(search_bar_hwnd);
    Sleep(20);
    AttachThreadInput(dwCurID, dwForeID, FALSE);
    const auto foreground = GetForegroundWindow();
    return foreground == search_bar_hwnd;
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
        std::thread checkTopWindow(checkTopWindowThread);
        checkTopWindow.detach();
        std::thread checkMouse(checkMouseThread);
        checkMouse.detach();
        DWORD dwTimeout = -1;
        SystemParametersInfo(SPI_GETFOREGROUNDLOCKTIMEOUT, 0, &dwTimeout, 0);
        if (dwTimeout >= 100)
        {
            SystemParametersInfo(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, nullptr, SPIF_SENDCHANGE | SPIF_UPDATEINIFILE);
        }
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
                    !(explorer_area.left <= point.x && point.x <= explorer_area.right &&
                        (explorer_area.top <= point.y && point.y <= explorer_area.bottom)) &&
                    !(search_bar_area.left <= point.x && point.x <= search_bar_area.right &&
                        (search_bar_area.top <= point.y && point.y <= search_bar_area.bottom));
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
    SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_UNAWARE);
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
