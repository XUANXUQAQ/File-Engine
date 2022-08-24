#pragma once
#include <string>

std::string to_utf8(const std::wstring& str);

std::string to_utf8(const wchar_t* buffer, int len);
