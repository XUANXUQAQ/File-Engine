#include <iostream>
#include "sqlite3.h"
#include <string>
#include <io.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <tchar.h>
#include <windows.h>
#include <locale>
#include <set>
//#define TEST

using namespace std;


set<string> command0;
set<string> command1;
set<string> command2;
set<string> command3;
set<string> command4;
set<string> command5;
set<string> command6;
set<string> command7;
set<string> command8;
set<string> command9;
set<string> command10;
set<string> command11;
set<string> command12;
set<string> command13;
set<string> command14;
set<string> command15;
set<string> command16;
set<string> command17;
set<string> command18;
set<string> command19;
set<string> command20;
set<string> command21;
set<string> command22;
set<string> command23;
set<string> command24;
set<string> command25;
vector<string> ignorePathVector;
int searchDepth;
char* searchPath;
sqlite3* db;
int dbRes;


void searchFiles(const char* path, const char* exd);
void addIgnorePath(const char* path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);

int getAscIISum(string name)
{
    int sum = 0;
    int length = name.length();
    for (int i = 0; i < length; i++)
    {
        if (name[i] > 0)
        {
            sum += name[i];
        }
    }
    return sum;
}

std::string to_utf8(const wchar_t* buffer, int len)
{
    int nChars = ::WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        NULL,
        0,
        NULL,
        NULL);
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
        NULL,
        NULL);

    return newbuffer;
}

std::string to_utf8(const std::wstring& str)
{
    return to_utf8(str.c_str(), (int)str.size());
}

std::wstring StringToWString(const std::string& str)
{
    setlocale(LC_ALL, "chs");
    const char* point_to_source = str.c_str();
    size_t new_size = str.size() + 1;
    wchar_t* point_to_destination = new wchar_t[new_size];
    wmemset(point_to_destination, 0, new_size);
    mbstowcs(point_to_destination, point_to_source, new_size);
    std::wstring result = point_to_destination;
    delete[] point_to_destination;
    setlocale(LC_ALL, "C");
    return result;
}

void saveResult(string path, int ascII)
{
    string sql;
    if (0 <= ascII && ascII <= 100)
    {
        sql = "INSERT OR IGNORE INTO list0  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command0.insert(sql);
    }
    else if (100 < ascII && ascII <= 200)
    {
        sql = "INSERT OR IGNORE INTO list1  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command1.insert(sql);
    }
    else if (200 < ascII && ascII <= 300)
    {
        sql = "INSERT OR IGNORE INTO list2  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command2.insert(sql);
    }
    else if (300 < ascII && ascII <= 400)
    {
        sql = "INSERT OR IGNORE INTO list3  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command3.insert(sql);
    }
    else if (400 < ascII && ascII <= 500)
    {
        sql = "INSERT OR IGNORE INTO list4  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command4.insert(sql);
    }
    else if (500 < ascII && ascII <= 600)
    {
        sql = "INSERT OR IGNORE INTO list5  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command5.insert(sql);
    }
    else if (600 < ascII && ascII <= 700)
    {
        sql = "INSERT OR IGNORE INTO list6  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command6.insert(sql);
    }
    else if (700 < ascII && ascII <= 800)
    {
        sql = "INSERT OR IGNORE INTO list7  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command7.insert(sql);
    }
    else if (800 < ascII && ascII <= 900)
    {
        sql = "INSERT OR IGNORE INTO list8  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command8.insert(sql);
    }
    else if (900 < ascII && ascII <= 1000)
    {
        sql = "INSERT OR IGNORE INTO list9  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command9.insert(sql);
    }
    else if (1000 < ascII && ascII <= 1100)
    {
        sql = "INSERT OR IGNORE INTO list10  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command10.insert(sql);
    }
    else if (1100 < ascII && ascII <= 1200)
    {
        sql = "INSERT OR IGNORE INTO list11  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command11.insert(sql);
    }
    else if (1200 < ascII && ascII <= 1300)
    {
        sql = "INSERT OR IGNORE INTO list12  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command12.insert(sql);
    }
    else if (1300 < ascII && ascII <= 1400)
    {
        sql = "INSERT OR IGNORE INTO list13  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command13.insert(sql);
    }
    else if (1400 < ascII && ascII <= 1500)
    {
        sql = "INSERT OR IGNORE INTO list14  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command14.insert(sql);
    }
    else if (1500 < ascII && ascII <= 1600)
    {
        sql = "INSERT OR IGNORE INTO list15  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command15.insert(sql);
    }
    else if (1600 < ascII && ascII <= 1700)
    {
        sql = "INSERT OR IGNORE INTO list16  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command16.insert(sql);
    }
    else if (1700 < ascII && ascII <= 1800)
    {
        sql = "INSERT OR IGNORE INTO list17  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command17.insert(sql);
    }
    else if (1800 < ascII && ascII <= 1900)
    {
        sql = "INSERT OR IGNORE INTO list18  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command18.insert(sql);
    }
    else if (1900 < ascII && ascII <= 2000)
    {
        sql = "INSERT OR IGNORE INTO list19  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command19.insert(sql);
    }
    else if (2000 < ascII && ascII <= 2100)
    {
        sql = "INSERT OR IGNORE INTO list20  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command20.insert(sql);
    }
    else if (2100 < ascII && ascII <= 2200)
    {
        sql = "INSERT OR IGNORE INTO list21  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command21.insert(sql);
    }
    else if (2200 < ascII && ascII <= 2300)
    {
        sql = "INSERT OR IGNORE INTO list22  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command22.insert(sql);
    }
    else if (2300 < ascII && ascII <= 2400)
    {
        sql = "INSERT OR IGNORE INTO list23  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command23.insert(sql);
    }
    else if (2400 < ascII && ascII <= 2500)
    {
        sql = "INSERT OR IGNORE INTO list24  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command24.insert(sql);
    }
    else
    {
        sql = "INSERT OR IGNORE INTO list25  VALUES (\"";
        sql.append(to_utf8(StringToWString(path))).append("\");");
        command25.insert(sql);
    }
}

void searchFilesIgnoreSearchDepth(const char* path, const char* exd)
{
    string file(path);
    string suffix(exd);
    cout << "start search without searchDepth" << endl;
    searchIgnoreSearchDepth(file, suffix);
    cout << "end search without searchDepth" << endl;
}

void searchIgnoreSearchDepth(string path, string exd)
{
    //cout << "getFiles()" << path<< endl;
    //文件句柄
    long hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
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
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    int ascII;

                    ascII = getAscIISum(name);

                    saveResult(_path, ascII);
                 
                    bool Ignore = isIgnore(path);
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
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    int ascII;

                    ascII = getAscIISum(name);
                    saveResult(_path, ascII);
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

void searchFiles(const char* path, const char* exd)
{
    string file(path);
    string suffix(exd);
    cout << "start Search" << endl;
    search(file, exd);
    cout << "end Search" << endl;
}

void addIgnorePath(const char* path)
{
    string str(path);
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    cout << "adding ignorePath:" << str << "..." << endl;
    ignorePathVector.push_back(str);
}

void setSearchDepth(int i)
{
    searchDepth = i;
}

int count(string path, string pattern)
{
    int begin = -1;
    int count = 0;
    while ((begin = path.find(pattern, (unsigned __int64)begin + 1)) != string::npos)
    {
        count++;
        begin = begin + pattern.length();
    }
    return count;
}

bool isSearchDepthOut(string path)
{
    int num = count(path, "\\");
    if (num > searchDepth - 2)
    {
        return true;
    }
    return false;
}

bool isIgnore(string path)
{
    if (path.find("$") != string::npos)
    {
        return true;
    }
    transform(path.begin(), path.end(), path.begin(), ::tolower);
    int size = ignorePathVector.size();
    for (int i = 0; i < size; i++)
    {
        if (path.find(ignorePathVector[i]) != string::npos)
        {
            return true;
        }
    }
    return false;
}

void search(string path, string exd)
{
    //cout << "getFiles()" << path<< endl;
    //文件句柄
    long long hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
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
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    int ascII;
                    ascII = getAscIISum(name);
                    saveResult(_path, ascII);
                    
                    bool SearchDepthOut = isSearchDepthOut(path);
                    bool Ignore = isIgnore(path);
                    bool result = !Ignore && !SearchDepthOut;
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
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    int ascII;

                    ascII = getAscIISum(name);
                    saveResult(_path, ascII);

                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

void executeAll(std::set<string>& command)
{
    std::set<string>::iterator iter;
    string _sql_command;
    char sql_command[1000];
    if (!command.empty())
    {
        sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
        for (iter = command.begin(); iter != command.end(); iter++)
        {
            _sql_command = *iter;
            strcpy_s(sql_command, 1000, _sql_command.c_str());
            sqlite3_exec(db, sql_command, NULL, NULL, NULL);
        }
        sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
    }
}

#ifndef TEST
int main(int argc, char* argv[])
{
    char searchPath[1000];
    char searchDepth[50];
    char ignorePath[3000];
    char output[1000];
    char isIgnoreSearchDepth[2];

    if (argc == 6)
    {
        strcpy_s(searchPath, argv[1]);
        strcpy_s(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy_s(ignorePath, argv[3]);
        strcpy_s(output, argv[4]);
        strcpy_s(isIgnoreSearchDepth, argv[5]);
     

        cout << "searchPath:" << searchPath << endl;
        cout << "searchDepth:" << searchDepth << endl;
        cout << "ignorePath:" << ignorePath << endl;
        cout << "output:" << output << endl;
        cout << "isIgnoreSearchDepth:" << isIgnoreSearchDepth << endl << endl;

        //打开数据库
        dbRes = sqlite3_open(output, &db);
        if (dbRes) {
            cout << "open database error" << endl << endl;
            exit(0);
        }
        else {
            cout << "open database successfully" << endl << endl;
        }

        sqlite3_exec(db, "PRAGMA synchronous = OFF; ", 0, 0, 0);

        char* p = NULL;
        char* _ignorepath = ignorePath;
        p = strtok(_ignorepath, ",");
        if (p != NULL)
        {
            addIgnorePath(p);
        }

        while (p != NULL)
        {
            p = strtok(NULL, ",");
            if (p != NULL)
            {
                addIgnorePath(p);
            }
        }

        cout << "ignorePathVector Size:" << ignorePathVector.size() << endl;
        if (atoi(isIgnoreSearchDepth) == 1)
        {
            cout << "ignore searchDepth!!!" << endl;
            searchFilesIgnoreSearchDepth(searchPath, "*");
        }

        else
        {
            cout << "normal search" << endl;
            searchFiles(searchPath, "*");
        }
        
        executeAll(command0);
        executeAll(command1);
        executeAll(command2);
        executeAll(command3);
        executeAll(command4);
        executeAll(command5);
        executeAll(command6);
        executeAll(command7);
        executeAll(command8);
        executeAll(command9);
        executeAll(command10);
        executeAll(command11);
        executeAll(command12);
        executeAll(command13);
        executeAll(command14);
        executeAll(command15);
        executeAll(command16);
        executeAll(command17);
        executeAll(command18);
        executeAll(command19);
        executeAll(command20);
        executeAll(command21);
        executeAll(command22);
        executeAll(command23);
        executeAll(command24);
        executeAll(command25);
        sqlite3_close(db);
        return 0;
    }
    else
    {
        cout << "args error" << endl;
    }
}
#else
int main()
{
    const char* output = "C:\\Users\\13927\\Desktop\\test.db";
    //打开数据库
    dbRes = sqlite3_open(output, &db);
    if (dbRes) {
        cout << "open database error" << endl << endl;
        exit(0);
    }
    else {
        cout << "open database successfully" << endl << endl;
    }

    sqlite3_exec(db, "PRAGMA synchronous = OFF; ", 0, 0, 0);

    setSearchDepth(6);
    addIgnorePath("C:\\Windows");
    searchFiles("C:", "*");
    executeAll(command0);
    executeAll(command1);
    executeAll(command2);
    executeAll(command3);
    executeAll(command4);
    executeAll(command5);
    executeAll(command6);
    executeAll(command7);
    executeAll(command8);
    executeAll(command9);
    executeAll(command10);
    executeAll(command11);
    executeAll(command12);
    executeAll(command13);
    executeAll(command14);
    executeAll(command15);
    executeAll(command16);
    executeAll(command17);
    executeAll(command18);
    executeAll(command19);
    executeAll(command20);
    executeAll(command21);
    executeAll(command22);
    executeAll(command23);
    executeAll(command24);
    executeAll(command25);
    sqlite3_close(db);
    getchar();
}

#endif
