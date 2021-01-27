#pragma once
#include <Windows.h>


bool is_search_bar_window(const HWND& hd);

bool is_explorer_window_low_cost(const HWND& hwnd);

bool is_explorer_window_high_cost(const HWND& hwnd);

bool is_file_chooser_window(const HWND& hwnd);

HWND getSearchBarHWND();