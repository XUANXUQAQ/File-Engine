#pragma once
#include <Windows.h>
#include <dwmapi.h>
#include <string>
#include <vector>

class DarkMode
{

public:
	static bool is_dark_theme()
	{
		// based on https://stackoverflow.com/questions/51334674/how-to-detect-windows-10-light-dark-mode-in-win32-application

		// The value is expected to be a REG_DWORD, which is a signed 32-bit little-endian
		auto buffer = std::vector<char>(4);
		auto cbData = static_cast<DWORD>(buffer.size() * sizeof(char));
		const auto res = RegGetValueW(
			HKEY_CURRENT_USER,
			L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
			L"AppsUseLightTheme",
			RRF_RT_REG_DWORD, // expected value type
			nullptr,
			buffer.data(),
			&cbData);

		if (res != ERROR_SUCCESS)
		{
			fprintf(stderr, "%s\n", ("Error: error_code=" + std::to_string(res)).c_str());
			return false;
		}

		// convert bytes written to our buffer to an int, assuming little-endian
		const auto i = buffer[3] << 24 |
			buffer[2] << 16 |
			buffer[1] << 8 |
			buffer[0];

		return i != 1;
	}
	DarkMode(const DarkMode&) = delete;
	DarkMode(DarkMode&&) = delete;
	DarkMode& operator=(const DarkMode&) = delete;
	DarkMode& operator=(DarkMode&&) = delete;
private:
	DarkMode() = default;
	~DarkMode() = default;
};
