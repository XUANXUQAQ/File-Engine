#pragma once

#include <string>
#include <algorithm>
#include <ctime>
#include <tchar.h>
#include <Windows.h>
#include <locale>
#include <iostream>
#include "sqlite3.h"
#include <string>
#include <io.h>
#include <fstream>
#include <vector>

using namespace std;

vector<string*> command0;
vector<string*> command1;
vector<string*> command2;
vector<string*> command3;
vector<string*> command4;
vector<string*> command5;
vector<string*> command6;
vector<string*> command7;
vector<string*> command8;
vector<string*> command9;
vector<string*> command10;
vector<string*> command11;
vector<string*> command12;
vector<string*> command13;
vector<string*> command14;
vector<string*> command15;
vector<string*> command16;
vector<string*> command17;
vector<string*> command18;
vector<string*> command19;
vector<string*> command20;
vector<string*> command21;
vector<string*> command22;
vector<string*> command23;
vector<string*> command24;
vector<string*> command25;
vector<string*> command26;
vector<string*> command27;
vector<string*> command28;
vector<string*> command29;
vector<string*> command30;
vector<string*> command31;
vector<string*> command32;
vector<string*> command33;
vector<string*> command34;
vector<string*> command35;
vector<string*> command36;
vector<string*> command37;
vector<string*> command38;
vector<string*> command39;
vector<string*> command40;
vector<string> ignorePathVector;

int searchDepth;
char* searchPath;
sqlite3* db;
int dbRes;


void searchFiles(const char* path, const char* exd);
void addIgnorePath(const char* path);
int count(const string& path, const string& pattern);
bool isSearchDepthOut(const string& path);
bool isIgnore(string path);
void search(const string& path, const string& exd);
void searchIgnoreSearchDepth(string path, string exd);
std::wstring StringToWString(const std::string& str);
void executeAll(vector<string*>& vec, const char* init);
void initAllVector();
std::string to_utf8(const std::wstring& str);
std::string to_utf8(const wchar_t* buffer, int len);
int getAscIISum(string name);


inline void initAllVector() {
    command0.reserve(30000);
    command1.reserve(30000);
    command2.reserve(30000);
    command3.reserve(30000);
    command4.reserve(30000);
    command5.reserve(30000);
    command6.reserve(30000);
    command7.reserve(30000);
    command8.reserve(30000);
    command9.reserve(30000);
    command10.reserve(30000);
    command11.reserve(30000);
    command12.reserve(30000);
    command13.reserve(30000);
    command14.reserve(30000);
    command15.reserve(30000);
    command16.reserve(30000);
    command17.reserve(30000);
    command18.reserve(30000);
    command19.reserve(30000);
    command20.reserve(30000);
    command21.reserve(30000);
    command22.reserve(30000);
    command23.reserve(30000);
    command24.reserve(30000);
    command25.reserve(30000);
    command26.reserve(30000);
    command27.reserve(30000);
    command28.reserve(30000);
    command29.reserve(30000);
    command30.reserve(30000);
    command31.reserve(30000);
    command32.reserve(30000);
    command33.reserve(30000);
    command34.reserve(30000);
    command35.reserve(30000);
    command36.reserve(30000);
    command37.reserve(30000);
    command38.reserve(30000);
    command39.reserve(30000);
    command40.reserve(30000);
}

inline void saveResult(const string path, const int ascII)
{
	auto* _path = new string(to_utf8(StringToWString(path)));
	const auto asciiGroup = ascII / 100;
    switch (asciiGroup)
    {
    case 0:
        command0.emplace_back(_path);
        break;

    case 1:
        command1.emplace_back(_path);
        break;

    case 2:
        command2.emplace_back(_path);
        break;

    case 3:
        command3.emplace_back(_path);
        break;

    case 4:
        command4.emplace_back(_path);
        break;

    case 5:
        command5.emplace_back(_path);
        break;
    case 6:
        command6.emplace_back(_path);
        break;

    case 7:
        command7.emplace_back(_path);
        break;

    case 8:
        command8.emplace_back(_path);
        break;

    case 9:
        command9.emplace_back(_path);
        break;

    case 10:
        command10.emplace_back(_path);
        break;

    case 11:
        command11.emplace_back(_path);
        break;

    case 12:
        command12.emplace_back(_path);
        break;

    case 13:
        command13.emplace_back(_path);
        break;

    case 14:
        command14.emplace_back(_path);
        break;

    case 15:
        command15.emplace_back(_path);
        break;

    case 16:
        command16.emplace_back(_path);
        break;

    case 17:
        command17.emplace_back(_path);
        break;

    case 18:
        command18.emplace_back(_path);
        break;

    case 19:
        command19.emplace_back(_path);
        break;

    case 20:
        command20.emplace_back(_path);
        break;

    case 21:
        command21.emplace_back(_path);
        break;

    case 22:
        command22.emplace_back(_path);
        break;

    case 23:
        command23.emplace_back(_path);
        break;

    case 24:
        command24.emplace_back(_path);
        break;

    case 25:
        command25.emplace_back(_path);
        break;

    case 26:
        command26.emplace_back(_path);
        break;
    case 27:
        command27.emplace_back(_path);
        break;
    case 28:
        command28.emplace_back(_path);
        break;
    case 29:
        command29.emplace_back(_path);
        break;
    case 30:
        command30.emplace_back(_path);
        break;
    case 31:
        command31.emplace_back(_path);
        break;
    case 32:
        command32.emplace_back(_path);
        break;
    case 33:
        command33.emplace_back(_path);
        break;
    case 34:
        command34.emplace_back(_path);
        break;
    case 35:
        command35.emplace_back(_path);
        break;
    case 36:
        command36.emplace_back(_path);
        break;
    case 37:
        command37.emplace_back(_path);
        break;
    case 38:
        command38.emplace_back(_path);
        break;
    case 39:
        command39.emplace_back(_path);
        break;
    case 40:
        command40.emplace_back(_path);
        break;

    default:
        break;
    }
}


inline void addIgnorePath(const char* path)
{
    string str(path);
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    cout << "adding ignorePath:" << str << "..." << endl;
    ignorePathVector.emplace_back(str);
}

inline void setSearchDepth(const int i)
{
    searchDepth = i;
}

inline auto count(const string& path, const string& pattern) -> int
{
    size_t begin = -1;
    auto count = 0;
    while ((begin = path.find(pattern, static_cast<unsigned long long>(begin) + 1)) != string::npos)
    {
        count++;
        begin = begin + pattern.length();
    }
    return count;
}

inline bool isSearchDepthOut(const string& path)
{
	const auto num = count(path, "\\");
    if (num > searchDepth - 2)
    {
        return true;
    }
    return false;
}

inline bool isIgnore(string path)
{
    if (path.find('$') != string::npos)
    {
        return true;
    }
    transform(path.begin(), path.end(), path.begin(), ::tolower);
    const auto size = ignorePathVector.size();
    for (size_t i = 0; i < size; i++)
    {
        if (path.find(ignorePathVector[i]) != string::npos)
        {
            return true;
        }
    }
    return false;
}

inline void search(const string& path, const string& exd)
{
    //cout << "getFiles()" << path<< endl;
    //文件句柄
    intptr_t hFile;
    //文件信息
    struct _finddata_t fileinfo{};
    string pathName, exdName;

    if (0 != strcmp(exd.c_str(), ""))
    {
        exdName = "\\*." + exd;
    }
    else
    {
        exdName = "\\*";
    }

    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            //cout << fileinfo.name << endl;

            //如果是文件夹中仍有文件夹,加入列表后迭代
            //如果不是,加入列表
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    const auto _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    const auto ascII = getAscIISum(name);
                    saveResult(_path, ascII);

                    const auto SearchDepthOut = isSearchDepthOut(path);
                    const auto Ignore = isIgnore(path);
                    const auto result = !Ignore && !SearchDepthOut;
                    if (result)
                    {
                        search(_path, exd);
                    }
                }
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    const auto _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);

                    int ascII = getAscIISum(name);
                    saveResult(_path, ascII);

                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

inline void executeAll(vector<string*>& vec, const char* init) {
    sqlite3_stmt* stmt = nullptr;
    string str;
    const size_t rc = sqlite3_prepare_v2(db, init, strlen(init), &stmt, 0);  // NOLINT(clang-diagnostic-shorten-64-to-32)
    if (rc != SQLITE_OK) {
        cout << "error preparing statement" << endl;
        cerr << "init sql:" << init << endl;
        exit(-1);
    }
    for (auto& iter : vec)
    {
        str = *iter;
        sqlite3_reset(stmt);
        const string::size_type iPos = str.find_last_of('\\') + 1;
        const auto filename = str.substr(iPos, str.length() - iPos);
        sqlite3_bind_int(stmt, 1, getAscIISum(filename));
        sqlite3_bind_text(stmt, 2, str.c_str(), -1, SQLITE_STATIC);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

inline void searchFiles(const char* path, const char* exd)
{
    cout << "start Search" << endl;
    const string file(path);
    const string suffix(exd);
    search(file, suffix);
    cout << "end Search" << endl;
}

inline void searchIgnoreSearchDepth(const string path, const string exd)
{
    //cout << "getFiles()" << path<< endl;
    //文件句柄
    intptr_t hFile;
    //文件信息
    struct _finddata_t fileinfo{};
    string pathName, exdName;

    if (0 != strcmp(exd.c_str(), ""))
    {
        exdName = "\\*." + exd;
    }
    else
    {
        exdName = "\\*";
    }

    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            //cout << fileinfo.name << endl;

            //如果是文件夹中仍有文件夹,加入列表后迭代
            //如果不是,加入列表
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    auto _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    int ascII;

                    ascII = getAscIISum(name);

                    saveResult(_path, ascII);

                    auto Ignore = isIgnore(path);
                    if (!Ignore)
                    {
                        searchIgnoreSearchDepth(_path, exd);
                        //cout << isResultReady << endl;
                    }
                }
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    auto _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);

                    auto ascII = getAscIISum(name);
                    saveResult(_path, ascII);
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}


inline void searchFilesIgnoreSearchDepth(const char* path, const char* exd)
{
	const string file(path);
    const string suffix(exd);
    cout << "start search without searchDepth" << endl;
    searchIgnoreSearchDepth(file, suffix);
    cout << "end search without searchDepth" << endl;
}

inline int getAscIISum(string name)
{
	auto sum = 0;
    const auto length = name.length();
    for (size_t i = 0; i < length; i++)
    {
        if (name[i] > 0)
        {
            sum += name[i];
        }
    }
    return sum;
}

inline std::string to_utf8(const wchar_t* buffer, int len)
{
    int nChars = ::WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (nChars == 0)
    {
        return "";
    }
    string newbuffer;
    newbuffer.resize(nChars);
    ::WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        const_cast<char*>(newbuffer.c_str()),
        nChars,
        nullptr,
        nullptr);

    return newbuffer;
}

inline std::string to_utf8(const std::wstring& str)
{
    return to_utf8(str.c_str(), static_cast<int>(str.size()));
}

inline std::wstring StringToWString(const std::string& str)
{
    setlocale(LC_ALL, "chs");
    const auto* const point_to_source = str.c_str();
    const auto new_size = str.size() + 1;
    auto* point_to_destination = new wchar_t[new_size];
    wmemset(point_to_destination, 0, new_size);
    mbstowcs(point_to_destination, point_to_source, new_size);
    std::wstring result = point_to_destination;
    delete[] point_to_destination;
    setlocale(LC_ALL, "C");
    return result;
}