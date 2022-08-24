#include <string>
#include <Windows.h>
#include "string_to_utf8.h"

std::string to_utf8(const std::wstring& str)
{
	return to_utf8(str.c_str(), static_cast<int>(str.size()));
}

std::string to_utf8(const wchar_t* buffer, int len)
{
	const auto nChars = WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		nullptr,
		0,
		nullptr,
		nullptr);
	if (nChars == 0)
	{
		return "";
	}
	std::string newBuffer;
	newBuffer.resize(nChars);
	WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		const_cast<char*>(newBuffer.c_str()),
		nChars,
		nullptr,
		nullptr);

	return newBuffer;
}
