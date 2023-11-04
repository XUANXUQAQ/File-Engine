#include <cstdio>
#include <Windows.h>
#include <shellscalingapi.h>
#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shcore.lib")

float get_dpi();

int main()
{
    const float dpi = get_dpi();
    printf("%f", dpi);
}

/**
 * 获取Windows缩放等级，适配高DPI
 */
float get_dpi()
{
    const auto hwnd = FindWindowA(nullptr, "File-Engine-SearchBar");
    const auto h_monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    UINT x, y; // x == y
    GetDpiForMonitor(h_monitor, MDT_EFFECTIVE_DPI, &x, &y);
    return static_cast<float>(x) / 96;
}
