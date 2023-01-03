#pragma once
#include <string>
#include <vector>
#include "sqlite3.h"

using parameter = struct parameter
{
    char disk{'\0'};
    std::vector<std::string> ignorePath;
    sqlite3* db{nullptr};
};

void init_usn(parameter p);
void split_string(const char* str, std::vector<std::string>& vec);
void init_tables(sqlite3* db);
