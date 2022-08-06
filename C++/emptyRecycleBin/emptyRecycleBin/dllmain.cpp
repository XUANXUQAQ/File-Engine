// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include "file_engine_dllInterface_EmptyRecycleBin.h"
#include <shellapi.h>
#pragma comment(lib, "Shell32")

void empty_recycle_bin();

JNIEXPORT void JNICALL Java_file_engine_dllInterface_EmptyRecycleBin_emptyRecycleBin
(JNIEnv*, jobject)
{
	empty_recycle_bin();
}

void empty_recycle_bin()
{
	SHEmptyRecycleBin(nullptr, nullptr, 0);
}
