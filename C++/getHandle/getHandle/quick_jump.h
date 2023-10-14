#pragma once
#include <Windows.h>
#include <Shlwapi.h>

#pragma comment(lib, "Shlwapi.lib")
// Deleter for a PIDL allocated by the shell.
struct CoTaskMemDeleter
{
    void operator()(ITEMIDLIST* pidl) const { CoTaskMemFree(pidl); }
};

void jump_to_dest(HWND hwnd, const wchar_t* path);
void set_file_selected(HWND hwnd, const wchar_t* file_name);