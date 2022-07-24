#pragma once
#include <Windows.h>


bool is_search_bar_window(const HWND& hd);

bool is_explorer_window_by_class_name(const HWND& hwnd);

bool is_explorer_window_by_process(const HWND& hwnd);

bool is_file_chooser_window(const HWND& hwnd);

HWND get_search_bar_hwnd();