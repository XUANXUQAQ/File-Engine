// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include "file_engine_dllInterface_GetWindowsKnownFolder.h"
#include <Windows.h>
#include <ShlObj.h>
#include <string>
#pragma comment(lib, "Shell32")
#pragma comment(lib, "Ole32")

std::wstring jstring2wstring(JNIEnv* env, jstring string);
jstring wstring2jstring(JNIEnv* env, const wchar_t* src);

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_GetWindowsKnownFolder_getKnownFolder
(JNIEnv* env, jobject, jstring guid_jstring)
{
	const auto&& guid_str = jstring2wstring(env, guid_jstring);
	GUID guid;
	if (FAILED(CLSIDFromString(guid_str.c_str(), &guid)))
	{
		return nullptr;
	}
	PWSTR path = nullptr;
	if (FAILED(SHGetKnownFolderPath(guid, 0, nullptr, &path)))
	{
		return nullptr;
	}
	CoTaskMemFree(path);
	return wstring2jstring(env, path);
}

std::wstring jstring2wstring(JNIEnv* env, jstring string)
{
	std::wstring value;
	const jchar* raw = env->GetStringChars(string, nullptr);
	const jsize len = env->GetStringLength(string);
	value.assign(raw, raw + len);
	env->ReleaseStringChars(string, raw);
	return value;
}

jstring wstring2jstring(JNIEnv* env, const wchar_t* src)
{
	const auto src_len = wcslen(src);
	const auto dest = new jchar[src_len + 1];
	memset(dest, 0, sizeof(jchar) * (src_len + 1));
	for (size_t i = 0; i < src_len; i++)
		memcpy(&dest[i], &src[i], 2);
	jstring dst = env->NewString(dest, static_cast<jsize>(src_len));
	delete[] dest;
	return dst;
}
