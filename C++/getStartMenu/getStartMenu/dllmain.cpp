// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <Windows.h>
#include <ShlObj.h>
#include <string>
#include "file_engine_dllInterface_GetStartMenu.h"
#pragma comment(lib, "Shell32")
#pragma comment(lib, "Ole32")
using namespace std;

wstring getStartMenu();
string wstringToString(const std::wstring& wstr);

/*
 * Class:     file_engine_dllInterface_GetStartMenu
 * Method:    getStartMenu
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_GetStartMenu_getStartMenu
(JNIEnv* env, jobject)
{
	const wstring tmp = getStartMenu();
	return env->NewStringUTF(wstringToString(tmp).c_str());
}

string wstringToString(const wstring& wstr)
{
	const LPCWSTR pwszSrc = wstr.c_str();
	const int nLen = WideCharToMultiByte(CP_ACP, 0, pwszSrc, -1, NULL, 0, NULL, NULL);
	if (nLen == 0)
		return "";

	auto pszDst = new char[nLen];
	if (!pszDst)
		return "";

	WideCharToMultiByte(CP_ACP, 0, pwszSrc, -1, pszDst, nLen, NULL, NULL);
	std::string str(pszDst);
	delete[] pszDst;
	pszDst = nullptr;

	return str;
}

wstring getStartMenu()
{
	PWSTR path = nullptr;
	static wstring _path;
	_path.clear();
	HRESULT hr = SHGetKnownFolderPath(FOLDERID_CommonStartMenu, 0, nullptr, &path);
	if (SUCCEEDED(hr))
	{
		_path.append(path);
	}
	CoTaskMemFree(path);
	_path.append(TEXT(";"));
	hr = SHGetKnownFolderPath(FOLDERID_StartMenu, 0, nullptr, &path);
	if (SUCCEEDED(hr))
	{
		_path.append(path);
	}
	CoTaskMemFree(path);
	return _path;
}
