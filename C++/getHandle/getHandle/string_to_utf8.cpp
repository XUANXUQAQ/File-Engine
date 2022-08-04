#include "pch.h"
#include <string>
#include <Windows.h>
#include "string_to_utf8.h"

std::string to_utf8(const std::wstring& str)
{
    return to_utf8(str.c_str(), static_cast<int>(str.size()));
}


std::string to_utf8(const wchar_t* buffer, int len)
{
    const auto n_chars = WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (n_chars == 0)
    {
        return "";
    }
    std::string new_buffer;
    new_buffer.resize(n_chars);
    WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        const_cast<char*>(new_buffer.c_str()),
        n_chars,
        nullptr,
        nullptr);

    return new_buffer;
}