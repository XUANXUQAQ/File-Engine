// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"

#include <chrono>
#include <Windows.h>
#include <WinUser.h>
#include "file_engine_dllInterface_HotkeyListener.h"
#include <fstream>
#pragma comment(lib, "User32.lib")


void registerHotKey(int key1, int key2, int key3, int key4, int key5);
bool getKeyStatus();
void startListen();
void stopListen();
inline time_t getCurrentMills();
inline int isVirtualKeyPressed(int vk);
void checkHotkeyAndAddCounter(int hotkey, unsigned* counter);

using DoubleClickKeyStateSaver = struct KeySaver
{
	int isKeyReleasedAfterPress = false;
	time_t keyPressedTime = 0;
};

static bool isKeyPressed = false;
static bool isStop = false;
static int hotkey1;
static int hotkey2;
static int hotkey3;
static int hotkey4;
static int hotkey5;
static bool isShiftDoubleClicked = false;
static bool isCtrlDoubleClicked = false;


JNIEXPORT void JNICALL Java_file_engine_dllInterface_HotkeyListener_registerHotKey
(JNIEnv*, jobject, jint key1, jint key2, jint key3, jint key4, jint key5)
{
	registerHotKey(key1, key2, key3, key4, key5);
}

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_HotkeyListener_getKeyStatus
(JNIEnv*, jobject)
{
	return getKeyStatus();
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

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_HotkeyListener_isCtrlDoubleClicked
(JNIEnv*, jobject)
{
	return isCtrlDoubleClicked;
}

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_HotkeyListener_isShiftDoubleClicked
(JNIEnv*, jobject)
{
	return isShiftDoubleClicked;
}

inline int isVirtualKeyPressed(int vk)
{
	return GetKeyState(vk) < 0;
}

void stopListen()
{
	isStop = true;
}

bool doubleClickCheck(int vk, DoubleClickKeyStateSaver& saver)
{
	bool ret = false;
	if (isVirtualKeyPressed(vk))
	{
		//ctrl 被点击
		if (getCurrentMills() - saver.keyPressedTime < 300 && saver.isKeyReleasedAfterPress)
		{
			ret = true;
		}
		else
		{
			saver.isKeyReleasedAfterPress = FALSE;
		}
		saver.keyPressedTime = getCurrentMills();
	}
	else
	{
		saver.isKeyReleasedAfterPress = getCurrentMills() - saver.keyPressedTime < 300;
	}
	return ret;
}

void startListen()
{
	short isKey1Pressed;
	short isKey2Pressed;
	short isKey3Pressed;
	short isKey4Pressed;
	short isKey5Pressed;
	DoubleClickKeyStateSaver ctrlSaver;
	DoubleClickKeyStateSaver shiftSaver;
	while (!isStop)
	{
		isCtrlDoubleClicked = doubleClickCheck(VK_CONTROL, ctrlSaver);
		isShiftDoubleClicked = doubleClickCheck(VK_SHIFT, shiftSaver);
		BYTE allKeyState[256]{ 0 };
		unsigned validHotkeyCount = 0;
		if (hotkey1 > 0)
		{
			isKey1Pressed = GetKeyState(hotkey1);
			checkHotkeyAndAddCounter(hotkey1, &validHotkeyCount);
		}
		else
		{
			isKey1Pressed = -1;
		}
		if (hotkey2 > 0)
		{
			isKey2Pressed = GetKeyState(hotkey2);
			checkHotkeyAndAddCounter(hotkey2, &validHotkeyCount);
		}
		else
		{
			isKey2Pressed = -1;
		}
		if (hotkey3 > 0)
		{
			isKey3Pressed = GetKeyState(hotkey3);
			checkHotkeyAndAddCounter(hotkey3, &validHotkeyCount);
		}
		else
		{
			isKey3Pressed = -1;
		}
		if (hotkey4 > 0)
		{
			isKey4Pressed = GetKeyState(hotkey4);
			checkHotkeyAndAddCounter(hotkey4, &validHotkeyCount);
		}
		else
		{
			isKey4Pressed = -1;
		}
		if (hotkey5 > 0)
		{
			isKey5Pressed = GetKeyState(hotkey5);
			checkHotkeyAndAddCounter(hotkey5, &validHotkeyCount);
		}
		else
		{
			isKey5Pressed = -1;
		}
		if (isKey1Pressed < 0 &&
			isKey2Pressed < 0 &&
			isKey3Pressed < 0 &&
			isKey4Pressed < 0 &&
			isKey5Pressed < 0) //如果某键被按下
		{
			unsigned pressedKeyCount = 0;
			GetKeyboardState(allKeyState);
			for (const unsigned char keyState : allKeyState)
			{
				if (keyState & 0x80)
				{
					++pressedKeyCount;
				}
				if (pressedKeyCount > validHotkeyCount)
				{
					break;
				}
			}
			if (pressedKeyCount == validHotkeyCount)
			{
				isKeyPressed = true;
			}
			else
			{
				isKeyPressed = false;
			}
		}
		else
		{
			isKeyPressed = false;
		}
		Sleep(1);
	}
}

void checkHotkeyAndAddCounter(int hotkey, unsigned* counter)
{
	if (hotkey == VK_CONTROL || hotkey == VK_MENU || hotkey == VK_SHIFT)
	{
		*counter += 2;
	}
	else
	{
		++* counter;
	}
}

bool getKeyStatus()
{
	return isKeyPressed;
}

void registerHotKey(const int key1, const int key2, const int key3, const int key4, const int key5)
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
		std::chrono::system_clock::now().time_since_epoch());
	return ms.count();
}
