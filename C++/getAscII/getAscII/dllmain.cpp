// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <cstring>
//#define TEST

using namespace std;
extern "C" __declspec(dllexport) int getAscII(const char* str);


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

__declspec(dllexport) int getAscII(const char* str)
{
    char _str[260];
    strcpy_s(_str, str);
    char* s = _str;
    int sum = 0;
    size_t length = strlen(_str);
    for (int i = 0; i < length; i++)
    {
        if (s[i] > 0)
        {
            sum += s[i];
        }
    }
    return sum;
}

#ifdef TEST
int main()
{
    int asc = getAscII("SANDISK备份");
    cout << asc << endl;
    getchar();
}
#endif
