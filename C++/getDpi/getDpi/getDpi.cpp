#include <cstdio>
#include <Windows.h>
#include <shellscalingapi.h>
#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shcore.lib")

void get_dpi();

int main()
{
    get_dpi();
}

/**
 * 获取Windows缩放等级，适配高DPI
 */
void get_dpi()
{
    const auto hwnd = FindWindowA(nullptr, "File-Engine-SearchBar");
    const auto h_monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    UINT x, y; // x == y
    GetDpiForMonitor(h_monitor, MDT_EFFECTIVE_DPI, &x, &y);
    printf("%f\n", static_cast<float>(x) / 96); // dpi

    MONITORINFO info;
    info.cbSize = sizeof(MONITORINFO);
    GetMonitorInfo(h_monitor, &info);
    const auto width = info.rcMonitor.right - info.rcMonitor.left;
    const auto height = info.rcMonitor.bottom - info.rcMonitor.top;
    printf("%ld\n", width);
    printf("%ld\n", height);
}
