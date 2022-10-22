// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include "darkmode.h"
#include "file_engine_dllInterface_SystemThemeInfo.h"
#pragma comment(lib, "Advapi32.lib")

/*
 * Class:     file_engine_dllInterface_SystemThemeInfo
 * Method:    isDarkThemeEnabled
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_SystemThemeInfo_isDarkThemeEnabled
(JNIEnv*, jobject)
{
	return DarkMode::is_dark_theme();
}