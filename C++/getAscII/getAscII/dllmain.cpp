// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <cstring>
#include "file_engine_dllInterface_GetAscII.h"

using namespace std;

int getAscII(const char* str);

JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetAscII_getAscII
(JNIEnv* env, jobject, jstring str)
{
    return getAscII(env->GetStringUTFChars(str, nullptr));
}

int getAscII(const char* str)
{
    const auto length = strlen(str);
    auto sum = 0;
    for (size_t i = 0; i < length; ++i)
    {
        if (str[i] > 0)
        {
            sum += str[i];
        }
    }
    return sum;
}
