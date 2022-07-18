#include "pch.h"

#include <algorithm>
#include <Windows.h>
#include <ShlObj.h>
#include <atlcomcli.h>  // for COM smart pointers
#include <iostream>
#include <vector>
#include <system_error>
#include <memory>
#pragma comment(lib, "shell32")
#pragma comment(lib, "Ole32")
#pragma comment(lib, "OleAut32")
// Throw a std::system_error if the HRESULT indicates failure.
template< typename T >
void throw_if_failed(HRESULT hr, T&& msg)
{
    if (FAILED(hr))
        throw std::system_error{ hr, std::system_category(), std::forward<T>(msg) };
}

// Deleter for a PIDL allocated by the shell.
struct CoTaskMemDeleter
{
    void operator()(ITEMIDLIST* pidl) const { ::CoTaskMemFree(pidl); }
};
// A smart pointer for PIDLs.
using UniquePidlPtr = std::unique_ptr< ITEMIDLIST, CoTaskMemDeleter >;

// Return value of GetCurrentExplorerFolders()
struct ExplorerFolderInfo
{
    HWND hwnd = nullptr;  // window handle of explorer
    UniquePidlPtr pidl;   // PIDL that points to current folder
};

// Get information about all currently open explorer windows.
// Throws std::system_error exception to report errors.
std::vector< ExplorerFolderInfo > GetCurrentExplorerFolders()
{
    CComPtr< IShellWindows > pshWindows;
    throw_if_failed(
        pshWindows.CoCreateInstance(CLSID_ShellWindows),
        "Could not create instance of IShellWindows");

    long count = 0;
    throw_if_failed(
        pshWindows->get_Count(&count),
        "Could not get number of shell windows");

    std::vector< ExplorerFolderInfo > result;
    result.reserve(count);

    for (long i = 0; i < count; ++i)
    {
        ExplorerFolderInfo info;

        CComVariant vi{ i };
        CComPtr< IDispatch > pDisp;
        throw_if_failed(
            pshWindows->Item(vi, &pDisp),
            "Could not get item from IShellWindows");

        if (!pDisp)
            // Skip - this shell window was registered with a NULL IDispatch
            continue;

        CComQIPtr< IWebBrowserApp > pApp{ pDisp };
        if (!pApp)
            // This window doesn't implement IWebBrowserApp 
            continue;

        // Get the window handle.
        pApp->get_HWND(reinterpret_cast<SHANDLE_PTR*>(&info.hwnd));

        CComQIPtr< IServiceProvider > psp{ pApp };
        if (!psp)
            // This window doesn't implement IServiceProvider
            continue;

        CComPtr< IShellBrowser > pBrowser;
        if (FAILED(psp->QueryService(SID_STopLevelBrowser, &pBrowser)))
            // This window doesn't provide IShellBrowser
            continue;

        CComPtr< IShellView > pShellView;
        if (FAILED(pBrowser->QueryActiveShellView(&pShellView)))
            // For some reason there is no active shell view
            continue;

        CComQIPtr< IFolderView > pFolderView{ pShellView };
        if (!pFolderView)
            // The shell view doesn't implement IFolderView
            continue;

        // Get the interface from which we can finally query the PIDL of
        // the current folder.
        CComPtr< IPersistFolder2 > pFolder;
        if (FAILED(pFolderView->GetFolder(IID_IPersistFolder2, reinterpret_cast<void**>(&pFolder))))
            continue;

        LPITEMIDLIST pidl = nullptr;
        if (SUCCEEDED(pFolder->GetCurFolder(&pidl)))
        {
            // Take ownership of the PIDL via std::unique_ptr.
            info.pidl = UniquePidlPtr{ pidl };
            result.push_back(std::move(info));
        }
    }
    return result;
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

std::string to_utf8(const std::wstring& str)
{
    return to_utf8(str.c_str(), static_cast<int>(str.size()));
}

/**
 * 获取explorer窗口当前显示的路径
 */
std::string getPathByHWND(const HWND& hwnd)
{
    if (!IsWindow(hwnd))
    {
        std::cout << "hwnd is not valid" << std::endl;
        return "";
    }
    //判断是否是桌面窗口句柄
    char windowClassName[260];
    GetClassNameA(hwnd, windowClassName, 260);
    std::string window_class_name_str(windowClassName);
    std::transform(window_class_name_str.begin(), window_class_name_str.end(), window_class_name_str.begin(), ::tolower);
    if (window_class_name_str.find("workerw") != std::string::npos)
    {
        //返回桌面位置
        WCHAR path[260];
        SHGetSpecialFolderPath(nullptr, path, CSIDL_DESKTOP, 0);
        return to_utf8(path);
    }
    try
    {
        throw_if_failed(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "failed to initialize com");
        wchar_t path[1000];
        for (const auto& info : GetCurrentExplorerFolders())
        {
            if (SHGetPathFromIDListEx(info.pidl.get(), path, ARRAYSIZE(path), 0))
            {
                if (info.hwnd == hwnd)
                {
                    const std::wstring path_wstr(path);
                    return to_utf8(path_wstr);
                }
                std::cout << "error, cannot find matched hwnd" << std::endl;
            }
        }
    }
    catch (std::system_error& e)
    {
        std::cerr << "ERROR: " << e.what() << "\nError code: " << e.code() << "\n";
    }
    CoUninitialize();
    return "";
}
