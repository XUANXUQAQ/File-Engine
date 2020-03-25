#include <iostream>
#include <windows.h>
#include <fstream>
//#define TEST

using namespace std;
extern "C" __declspec(dllexport) void registerHotKey(int key1, int key2, int key3);
extern "C" __declspec(dllexport) bool getKeyStatus();
extern "C" __declspec(dllexport) void startListen();
extern "C" __declspec(dllexport) void stopListen();
bool isExist(const char *FileName);

static bool isKeyPressed = false;
static bool isStop = false;
static int hotkey1;
static int hotkey2;
static int hotkey3;

void stopListen()
{
    isStop = true;
}

void startListen()
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

bool getKeyStatus()
{
    return isKeyPressed;
}
void registerHotKey(int key1, int key2, int key3)
{
    hotkey1 = key1;
    hotkey2 = key2;
    hotkey3 = key3;
}

bool isExist(const char *FileName)
{
    char FILENAME[600];
    strcpy(FILENAME, FileName);
    fstream _file;
    _file.open(FILENAME, ios::in);
    if (!_file)
    {
        return false;
    }
    else
    {
        return true;
    }
}

#ifdef TEST
int main()
{
    registerHotKey(17, -1, 74);
    startListen();
}
#endif