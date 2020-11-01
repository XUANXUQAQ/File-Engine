#pragma once
#include <Windows.h>


bool isSearchBarWindow(const HWND& hd);

bool isExplorerWindowLowCost(const HWND& hwnd);

bool isExplorerWindowHighCost(const HWND& hwnd);

bool isFileChooserWindow(const HWND& hwnd);