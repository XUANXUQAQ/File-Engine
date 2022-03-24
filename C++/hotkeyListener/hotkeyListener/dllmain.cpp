// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"

#include <chrono>
#include <Windows.h>
#include <WinUser.h>
#include "file_engine_dllInterface_HotkeyListener.h"
#include <fstream>
#pragma comment(lib, "User32.lib")

using namespace std;

extern "C" __declspec(dllexport) void registerHotKey(int key1, int key2, int key3, int key4, int key5);
extern "C" __declspec(dllexport) BOOL getKeyStatus();
extern "C" __declspec(dllexport) void startListen();
extern "C" __declspec(dllexport) void stopListen();
extern "C" __declspec(dllexport) void setCtrlDoubleClick(BOOL isResponse);

inline time_t getCurrentMills();
inline int isVirtualKeyPressed(int vk);

static volatile BOOL isKeyPressed = FALSE;
static volatile bool isStop = false;
static volatile int hotkey1;
static volatile int hotkey2;
static volatile int hotkey3;
static volatile int hotkey4;
static volatile int hotkey5;
static volatile time_t ctrlPressedTime;

bool isResponseCtrlDoubleClick = true;

JNIEXPORT void JNICALL Java_file_engine_dllInterface_HotkeyListener_registerHotKey
(JNIEnv*, jobject, jint key1, jint key2, jint key3, jint key4, jint key5)
{
	registerHotKey(key1, key2, key3, key4, key5);
}

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_HotkeyListener_getKeyStatus
(JNIEnv*, jobject)
{
	return static_cast<jboolean>(getKeyStatus());
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_HotkeyListener_startListen
(JNIEnv*, jobject)
{
	startListen();
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_HotkeyListener_stopListen
(JNIEnv*, jobject)
{
	stopListen();
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_HotkeyListener_setCtrlDoubleClick
(JNIEnv*, jobject, jboolean val)
{
	setCtrlDoubleClick(val);
}

inline int isVirtualKeyPressed(int vk)
{
	return GetKeyState(vk) < 0;
}

void setCtrlDoubleClick(BOOL isResponse)
{
	isResponseCtrlDoubleClick = isResponse;
}

__declspec(dllexport) void stopListen()
{
	isStop = true;
}

__declspec(dllexport) void startListen()
{
	short isKey1Pressed;
	short isKey2Pressed;
	short isKey3Pressed;
	short isKey4Pressed;
	short isKey5Pressed;
	short isCtrlPressedDouble = FALSE;
	int isCtrlReleasedAfterPress = FALSE;
	ctrlPressedTime = 0;

	while (!isStop)
	{
		if (isResponseCtrlDoubleClick)
		{
			isCtrlPressedDouble = false;
			if (isVirtualKeyPressed(VK_CONTROL))
			{
				//ctrl 被点击
				if (getCurrentMills() - ctrlPressedTime < 300 && isCtrlReleasedAfterPress)
				{
					isCtrlPressedDouble = TRUE;
				}
				else
				{
					isCtrlReleasedAfterPress = FALSE;
				}
				ctrlPressedTime = getCurrentMills();
			}
			else
			{
				if (getCurrentMills() - ctrlPressedTime < 300)
				{
					isCtrlReleasedAfterPress = TRUE;
				}
				else
				{
					isCtrlReleasedAfterPress = FALSE;
				}
			}
		}
		if (hotkey1 > 0)
		{
			isKey1Pressed = GetKeyState(hotkey1);
		}
		else
		{
			isKey1Pressed = -1;
		}
		if (hotkey2 > 0)
		{
			isKey2Pressed = GetKeyState(hotkey2);
		}
		else
		{
			isKey2Pressed = -1;
		}
		if (hotkey3 > 0)
		{
			isKey3Pressed = GetKeyState(hotkey3);
		}
		else
		{
			isKey3Pressed = -1;
		}
		if (hotkey4 > 0)
		{
			isKey4Pressed = GetKeyState(hotkey4);
		}
		else
		{
			isKey4Pressed = -1;
		}
		if (hotkey5 > 0)
		{
			isKey5Pressed = GetKeyState(hotkey5);
		}
		else
		{
			isKey5Pressed = -1;
		}
		if (isCtrlPressedDouble || isKey1Pressed < 0 && isKey2Pressed < 0 && isKey3Pressed < 0 && isKey4Pressed < 0 &&
			isKey5Pressed < 0) //如果某键被按下
		{
			isKeyPressed = TRUE;
		}
		else
		{
			isKeyPressed = FALSE;
		}
		Sleep(10);
	}
}

__declspec(dllexport) BOOL getKeyStatus()
{
	return isKeyPressed;
}

__declspec(dllexport) void registerHotKey(int key1, int key2, int key3, int key4, int key5)
{
	hotkey1 = key1;
	hotkey2 = key2;
	hotkey3 = key3;
	hotkey4 = key4;
	hotkey5 = key5;
}

inline time_t getCurrentMills()
{
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		chrono::system_clock::now().time_since_epoch());
	return ms.count();
}
