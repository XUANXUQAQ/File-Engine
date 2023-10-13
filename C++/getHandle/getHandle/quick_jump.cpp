#include "pch.h"
#include "quick_jump.h"
#include <system_error>
#include <memory>
#include <ExDisp.h>
#include <ShlObj.h>
#include <atlcomcli.h>  // for COM smart pointers

#include "checkHwnd.h"

// Throw a std::system_error if the HRESULT indicates failure.
template <typename T>
void throw_if_failed(HRESULT hr, T&& msg)
{
    if (FAILED(hr))
        throw std::system_error{hr, std::system_category(), std::forward<T>(msg)};
}

IShellBrowser* GetShellBrowserFromHWND(HWND hwnd)
{
    CComPtr<IShellWindows> pshWindows;
    IShellBrowser* pShellBrowser = nullptr;
    throw_if_failed(
        pshWindows.CoCreateInstance(CLSID_ShellWindows),
        "Could not create instance of IShellWindows");

    long count = 0;
    throw_if_failed(
        pshWindows->get_Count(&count),
        "Could not get number of shell windows");
    HWND temp_hwnd = nullptr;
    for (long i = 0; i < count; ++i)
    {
        CComVariant vi{i};
        CComPtr<IDispatch> pDisp;
        throw_if_failed(
            pshWindows->Item(vi, &pDisp),
            "Could not get item from IShellWindows");

        if (!pDisp)
            // Skip - this shell window was registered with a NULL IDispatch
            continue;

        CComQIPtr<IWebBrowserApp> pApp{pDisp};
        if (!pApp)
            // This window doesn't implement IWebBrowserApp
            continue;

        // Get the window handle.
        pApp->get_HWND(reinterpret_cast<SHANDLE_PTR*>(&temp_hwnd));
        if (temp_hwnd == hwnd)
        {
            throw_if_failed(IUnknown_QueryService(pApp.p, SID_STopLevelBrowser, IID_PPV_ARGS(&pShellBrowser)),
                            "Failed to get shell browser");

            // throw_if_failed(pApp->QueryInterface(IID_IShellBrowser, reinterpret_cast<void**>(&pShellBrowser)), "Failed to get shell browser");
            break;
        }
    }
    return pShellBrowser;
}

void jump_to_dest(HWND hwnd, const wchar_t* path)
{
    if (!is_explorer_window_by_class_name(hwnd))
    {
        return;
    }
    CoInitialize(nullptr);
    // Replace 'yourHwnd' with the HWND you want to obtain IShellBrowser for.

    IShellBrowser* pShellBrowser = GetShellBrowserFromHWND(hwnd);

    if (pShellBrowser != nullptr)
    {
        LPITEMIDLIST pidl = ILCreateFromPath(path);
        pShellBrowser->BrowseObject(pidl, SBSP_SAMEBROWSER);
        ILFree(pidl);
        pShellBrowser->Release();
    }
    CoUninitialize();
}
