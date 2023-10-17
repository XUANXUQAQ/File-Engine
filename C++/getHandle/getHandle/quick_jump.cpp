#include "pch.h"
#include "quick_jump.h"
#include <system_error>
#include <memory>
#include <ExDisp.h>
#include <ShlObj.h>
#include <atlcomcli.h>  // for COM smart pointers
#include <Propkey.h>
#include "checkHwnd.h"
#include "string_to_utf8.h"

// Throw a std::system_error if the HRESULT indicates failure.
template <typename T>
void throw_if_failed(HRESULT hr, T&& msg)
{
    if (FAILED(hr))
        throw std::system_error{hr, std::system_category(), std::forward<T>(msg)};
}

IFolderView2* GetFolderViewFromHWND(HWND hwnd)
{
    CComPtr<IShellWindows> pshWindows;
    IFolderView2* pfv = nullptr;
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
            IShellBrowser* pShellBrowser;
            throw_if_failed(IUnknown_QueryService(pApp.p, SID_STopLevelBrowser, IID_PPV_ARGS(&pShellBrowser)),
                            "Failed to get shell browser");
            IShellView* pShellView;
            throw_if_failed(pShellBrowser->QueryActiveShellView(&pShellView), "");

            throw_if_failed(pShellView->QueryInterface(IID_IFolderView2, reinterpret_cast<void**>(&pfv)), "");
            pShellView->Release();
            pShellBrowser->Release();
            break;
        }
    }
    return pfv;
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
            break;
        }
    }
    return pShellBrowser;
}

void jump_to_dest(HWND hwnd, const wchar_t* path)
{
    if (is_explorer_window_by_class_name(hwnd))
    {
        IShellBrowser* pShellBrowser = GetShellBrowserFromHWND(hwnd);

        if (pShellBrowser != nullptr)
        {
            LPITEMIDLIST pidl = ILCreateFromPath(path);
            pShellBrowser->BrowseObject(pidl, SBSP_SAMEBROWSER);
            SHChangeNotify(SHCNE_ALLEVENTS, SHCNF_FLUSH, nullptr, nullptr);
            ILFree(pidl);
            pShellBrowser->Release();
        }
    }
    else if (is_file_chooser_window(hwnd))
    {
        HWND hCtl = GetDlgItem(hwnd, 0x47C); // Combo
        if (!hCtl) hCtl = GetDlgItem(hwnd, 0x480); // Old Edit
        if (hCtl && *path && static_cast<int>(SendMessage(hCtl, WM_SETTEXT, 0, reinterpret_cast<SIZE_T>(path))) > 0)
        {
            const DWORD dwCurID = GetCurrentThreadId();
            const DWORD dwForeID = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
            AttachThreadInput(dwCurID, dwForeID, TRUE);
            SetForegroundWindow(hwnd);
            AttachThreadInput(dwCurID, dwForeID, FALSE);
            INPUT input{};
            input.type = INPUT_KEYBOARD;
            input.ki.wVk = VK_RETURN;
            SendInput(1, &input, sizeof(INPUT));
        }
    }
}

void set_file_selected(HWND hwnd, const wchar_t* file_name)
{
    if (is_explorer_window_by_class_name(hwnd))
    {
        IFolderView2* pFolderView = GetFolderViewFromHWND(hwnd);
        if (pFolderView != nullptr)
        {
            int count = 0;
            throw_if_failed(pFolderView->ItemCount(SVGIO_ALLVIEW, &count), "Get directory item count failed");
            for (int i = 0; i < count; ++i)
            {
                IShellItem2* pShellItem;
                throw_if_failed(pFolderView->GetItem(i, IID_IShellItem2, reinterpret_cast<void**>(&pShellItem)),
                                "Get shell item failed");
                LPWSTR file_name_view = nullptr;
                throw_if_failed(pShellItem->GetString(PKEY_FileName, &file_name_view),
                                "Get item display name failed");
                const auto cmp_res = wcscmp(file_name, file_name_view);
                CoTaskMemFree(file_name_view);
                pShellItem->Release();
                if (cmp_res == 0)
                {
                    pFolderView->SelectItem(i, SVSI_SELECT | SVSI_DESELECTOTHERS);
                    break;
                }
            }
            pFolderView->Release();
        }
    }
}
