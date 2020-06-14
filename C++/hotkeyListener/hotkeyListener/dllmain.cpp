// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
#include <WinUser.h>
#include <fstream>
#pragma comment(lib, "User32.lib")
//#define TEST

using namespace std;
extern "C" __declspec(dllexport) void registerHotKey(int key1, int key2, int key3);
extern "C" __declspec(dllexport) bool getKeyStatus();
extern "C" __declspec(dllexport) void startListen();
extern "C" __declspec(dllexport) void stopListen();

static volatile bool isKeyPressed = false;
static volatile bool isStop = false;
static volatile int hotkey1;
static volatile int hotkey2;
static volatile int hotkey3;

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
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
    while (!isStop)
    {
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

        if (isKey1Pressed < 0 && isKey2Pressed < 0 && isKey3Pressed < 0) //如果某键被按下
        {
            isKeyPressed = true;
#ifdef TEST
            cout << "key pressed" << endl;
#endif
        }
        else
        {
            isKeyPressed = false;
        }
        Sleep(10);
    }
}

__declspec(dllexport) bool getKeyStatus()
{
    return isKeyPressed;
}
__declspec(dllexport) void registerHotKey(int key1, int key2, int key3)
{
    hotkey1 = key1;
    hotkey2 = key2;
    hotkey3 = key3;
}

#ifdef TEST
int main()
{
    registerHotKey(17, -1, 74);
    startListen();
}
#endif

