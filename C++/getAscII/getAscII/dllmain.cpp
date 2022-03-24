// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <cstring>
#include "file_engine_dllInterface_GetAscII.h"
//#define TEST

using namespace std;
extern "C" __declspec(dllexport) int getAscII(const char* str);

JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetAscII_getAscII
(JNIEnv* env, jobject, jstring str)
{
    return getAscII(env->GetStringUTFChars(str, nullptr));
}

__declspec(dllexport) int getAscII(const char* str)
{
    char _str[260];
    strcpy_s(_str, str);
    auto* s = _str;
    auto sum = 0;
    const auto length = strlen(_str);
    for (auto i = 0; i < length; i++)
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
