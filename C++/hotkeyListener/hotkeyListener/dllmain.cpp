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
void checkHotkeyAndAddCounter(int hotkey, unsigned& counter);
bool getKeyPressedStatus(const int& hotkey);
unsigned getTotalKeyPressedCount();

using DoubleClickKeyStateSaver = struct KeySaver
{
	bool isKeyReleasedAfterPress = false;
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
static unsigned validHotkeyCount = 0;


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

bool doubleClickCheck(const int vk, DoubleClickKeyStateSaver& saver)
{
	bool ret = false;
	const auto keyPressed = getTotalKeyPressedCount();
	if (isVirtualKeyPressed(vk) && keyPressed == 2)
	{
		if (getCurrentMills() - saver.keyPressedTime < 300 && saver.isKeyReleasedAfterPress)
		{
			ret = true;
		}
		else
		{
			saver.isKeyReleasedAfterPress = false;
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
	DoubleClickKeyStateSaver ctrlSaver;
	DoubleClickKeyStateSaver shiftSaver;
	while (!isStop)
	{
		isCtrlDoubleClicked = doubleClickCheck(VK_CONTROL, ctrlSaver);
		isShiftDoubleClicked = doubleClickCheck(VK_SHIFT, shiftSaver);
		const bool isKey1Pressed = getKeyPressedStatus(hotkey1);
		const bool isKey2Pressed = getKeyPressedStatus(hotkey2);
		const bool isKey3Pressed = getKeyPressedStatus(hotkey3);
		const bool isKey4Pressed = getKeyPressedStatus(hotkey4);
		const bool isKey5Pressed = getKeyPressedStatus(hotkey5);
		const auto pressedKeyCount = getTotalKeyPressedCount();
		if (isKey1Pressed &&
			isKey2Pressed &&
			isKey3Pressed &&
			isKey4Pressed &&
			isKey5Pressed &&
			pressedKeyCount == validHotkeyCount) //如果某键被按下
		{
			isKeyPressed = true;
		}
		else
		{
			isKeyPressed = false;
		}
		Sleep(1);
	}
}

unsigned getTotalKeyPressedCount()
{
	BYTE allKeyState[256]{ 0 };
	GetKeyboardState(allKeyState);

	unsigned pressedKeyCount = 0;
	for (unsigned i = 0; i <= sizeof allKeyState; ++i)
	{
		const BYTE keyState = allKeyState[i];
		if ((0x30 <= i && i <= 0x5A) ||
			(VK_F1 <= i && i <= VK_F12) ||
			i == VK_SHIFT || i == VK_LSHIFT || i == VK_RSHIFT ||
			i == VK_CONTROL || i == VK_LCONTROL || i == VK_RCONTROL ||
			i == VK_MENU || i == VK_LMENU || i == VK_RMENU)
		{
			if (keyState & static_cast<BYTE>(0x80))
			{
				++pressedKeyCount;
			}
		}
	}
	return pressedKeyCount;
}

bool getKeyPressedStatus(const int& hotkey)
{
	if (hotkey > 0)
	{
		return GetKeyState(hotkey) & 0x80;
	}
	return true;
}

void checkHotkeyAndAddCounter(int hotkey, unsigned& counter)
{
	if (hotkey < 0)
	{
		return;
	}
	if (hotkey == VK_CONTROL || hotkey == VK_MENU || hotkey == VK_SHIFT)
	{
		counter += 2;
	}
	else
	{
		++counter;
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
	validHotkeyCount = 0;
	checkHotkeyAndAddCounter(key1, validHotkeyCount);
	checkHotkeyAndAddCounter(key2, validHotkeyCount);
	checkHotkeyAndAddCounter(key3, validHotkeyCount);
	checkHotkeyAndAddCounter(key4, validHotkeyCount);
	checkHotkeyAndAddCounter(key5, validHotkeyCount);
}

inline time_t getCurrentMills()
{
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch());
	return ms.count();
}
