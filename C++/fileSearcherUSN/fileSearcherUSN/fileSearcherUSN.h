#pragma once
#include <thread>
#include "search.h"

typedef struct PARAMETER
{
	char disk{'\0'};
	std::vector<std::string> ignorePath;
	sqlite3* db{nullptr};
} parameter;


void initUSN(parameter p);
void splitString(const char* str, std::vector<std::string>& vec);
void initTables(sqlite3* db);
