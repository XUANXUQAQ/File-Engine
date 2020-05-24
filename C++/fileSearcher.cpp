#include <string>
#include <io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <tchar.h>
#include <windows.h>
#include <locale>

using namespace std;
//#define TEST

vector<string> ignorePathVector;
int searchDepth;
char *searchPath;
ofstream results0;
ofstream results1;
ofstream results2;
ofstream results3;
ofstream results4;
ofstream results5;
ofstream results6;
ofstream results7;
ofstream results8;
ofstream results9;
ofstream results10;
ofstream results11;
ofstream results12;
ofstream results13;
ofstream results14;
ofstream results15;
ofstream results16;
ofstream results17;
ofstream results18;
ofstream results19;
ofstream results20;
ofstream results21;
ofstream results22;
ofstream results23;
ofstream results24;
ofstream results25;

void searchFiles(const char *path, const char *exd);
void addIgnorePath(const char *path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);
void clearResults();

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

std::string to_utf8(const wchar_t *buffer, int len)
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
        const_cast<char *>(newbuffer.c_str()),
        nChars,
        NULL,
        NULL);

    return newbuffer;
}

std::string to_utf8(const std::wstring &str)
{
    return to_utf8(str.c_str(), (int)str.size());
}

std::wstring StringToWString(const std::string &str)
{
    setlocale(LC_ALL, "chs");
    const char *point_to_source = str.c_str();
    size_t new_size = str.size() + 1;
    wchar_t *point_to_destination = new wchar_t[new_size];
    wmemset(point_to_destination, 0, new_size);
    mbstowcs(point_to_destination, point_to_source, new_size);
    std::wstring result = point_to_destination;
    delete[] point_to_destination;
    setlocale(LC_ALL, "C");
    return result;
}

void saveResult(string path, int ascII)
{
#ifndef TEST
    if (0 < ascII && ascII <= 100)
    {
        results0 << to_utf8(StringToWString(path));
        results0 << ("\n");
    }
    else if (100 < ascII && ascII <= 200)
    {
        results1 << to_utf8(StringToWString(path));
        results1 << ("\n");
    }
    else if (200 < ascII && ascII <= 300)
    {
        results2 << to_utf8(StringToWString(path));
        results2 << ("\n");
    }
    else if (300 < ascII && ascII <= 400)
    {
        results3 << to_utf8(StringToWString(path));
        results3 << ("\n");
    }
    else if (400 < ascII && ascII <= 500)
    {
        results4 << to_utf8(StringToWString(path));
        results4 << ("\n");
    }
    else if (500 < ascII && ascII <= 600)
    {
        results5 << to_utf8(StringToWString(path));
        results5 << ("\n");
    }
    else if (600 < ascII && ascII <= 700)
    {
        results6 << to_utf8(StringToWString(path));
        results6 << ("\n");
    }
    else if (700 < ascII && ascII <= 800)
    {
        results7 << to_utf8(StringToWString(path));
        results7 << ("\n");
    }
    else if (800 < ascII && ascII <= 900)
    {
        results8 << to_utf8(StringToWString(path));
        results8 << ("\n");
    }
    else if (900 < ascII && ascII <= 1000)
    {
        results9 << to_utf8(StringToWString(path));
        results9 << ("\n");
    }
    else if (1000 < ascII && ascII <= 1100)
    {
        results10 << to_utf8(StringToWString(path));
        results10 << ("\n");
    }
    else if (1100 < ascII && ascII <= 1200)
    {
        results11 << to_utf8(StringToWString(path));
        results11 << ("\n");
    }
    else if (1200 < ascII && ascII <= 1300)
    {
        results12 << to_utf8(StringToWString(path));
        results12 << ("\n");
    }
    else if (1300 < ascII && ascII <= 1400)
    {
        results13 << to_utf8(StringToWString(path));
        results13 << ("\n");
    }
    else if (1400 < ascII && ascII <= 1500)
    {
        results14 << to_utf8(StringToWString(path));
        results14 << ("\n");
    }
    else if (1500 < ascII && ascII <= 1600)
    {
        results15 << to_utf8(StringToWString(path));
        results15 << ("\n");
    }
    else if (1600 < ascII && ascII <= 1700)
    {
        results16 << to_utf8(StringToWString(path));
        results16 << ("\n");
    }
    else if (1700 < ascII && ascII <= 1800)
    {
        results17 << to_utf8(StringToWString(path));
        results17 << ("\n");
    }
    else if (1800 < ascII && ascII <= 1900)
    {
        results18 << to_utf8(StringToWString(path));
        results18 << ("\n");
    }
    else if (1900 < ascII && ascII <= 2000)
    {
        results19 << to_utf8(StringToWString(path));
        results19 << ("\n");
    }
    else if (2000 < ascII && ascII <= 2100)
    {
        results20 << to_utf8(StringToWString(path));
        results20 << ("\n");
    }
    else if (2100 < ascII && ascII <= 2200)
    {
        results21 << to_utf8(StringToWString(path));
        results21 << ("\n");
    }
    else if (2200 < ascII && ascII <= 2300)
    {
        results22 << to_utf8(StringToWString(path));
        results22 << ("\n");
    }
    else if (2300 < ascII && ascII <= 2400)
    {
        results23 << to_utf8(StringToWString(path));
        results23 << ("\n");
    }
    else if (2400 < ascII && ascII <= 2500)
    {
        results24 << to_utf8(StringToWString(path));
        results24 << ("\n");
    }
    else
    {
        results25 << to_utf8(StringToWString(path));
        results25 << ("\n");
    }
#else
    wstring record = to_utf8(StringToWString(path));
    results0 << record << endl;
#endif
}

void searchFilesIgnoreSearchDepth(const char *path, const char *exd)
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

                    if (ascII > 2500)
                    {
                        saveResult(_path, ascII);
                    }
                    else if (ascII <= 2500)
                    {
                        saveResult(_path, ascII);
                    }

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

                    if (ascII > 2500)
                    {
                        saveResult(_path, ascII);
                    }
                    else if (ascII <= 2500)
                    {
                        saveResult(_path, ascII);
                    }
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

void searchFiles(const char *path, const char *exd)
{
    string file(path);
    string suffix(exd);
    cout << "start Search" << endl;
    search(file, exd);
    cout << "end Search" << endl;
}

void addIgnorePath(const char *path)
{
    string str(path);
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    cout << "adding ignorePath:" << str << endl;
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
    while ((begin = path.find(pattern, begin + 1)) != string::npos)
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
                    if (ascII > 2500)
                    {
                        saveResult(_path, ascII);
                    }
                    else if (ascII <= 2500)
                    {
                        saveResult(_path, ascII);
                    }

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

                    if (ascII > 2500)
                    {
                        saveResult(_path, ascII);
                    }
                    else if (ascII <= 2500)
                    {
                        saveResult(_path, ascII);
                    }
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

#ifndef TEST
int main(int argc, char *argv[])
{
    char searchPath[260];
    char searchDepth[50];
    char ignorePath[3000];
    char output[260];
    char isIgnoreSearchDepth[2];
    char output0_100[260];
    char output100_200[260];
    char output200_300[260];
    char output300_400[260];
    char output400_500[260];
    char output500_600[260];
    char output600_700[260];
    char output700_800[260];
    char output800_900[260];
    char output900_1000[260];
    char output1000_1100[260];
    char output1100_1200[260];
    char output1200_1300[260];
    char output1300_1400[260];
    char output1400_1500[260];
    char output1500_1600[260];
    char output1600_1700[260];
    char output1700_1800[260];
    char output1800_1900[260];
    char output1900_2000[260];
    char output2000_2100[260];
    char output2100_2200[260];
    char output2200_2300[260];
    char output2300_2400[260];
    char output2400_2500[260];
    char output2500_[260];

    if (argc == 6)
    {
        strcpy(searchPath, argv[1]);
        strcpy(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy(ignorePath, argv[3]);
        strcpy(output, argv[4]);
        strcpy(isIgnoreSearchDepth, argv[5]);
        strcpy(output0_100, output);
        strcat(output0_100, "\\list0-100.txt");
        strcpy(output100_200, output);
        strcat(output100_200, "\\list100-200.txt");
        strcpy(output200_300, output);
        strcat(output200_300, "\\list200-300.txt");
        strcpy(output300_400, output);
        strcat(output300_400, "\\list300-400.txt");
        strcpy(output400_500, output);
        strcat(output400_500, "\\list400-500.txt");
        strcpy(output500_600, output);
        strcat(output500_600, "\\list500-600.txt");
        strcpy(output600_700, output);
        strcat(output600_700, "\\list600-700.txt");
        strcpy(output700_800, output);
        strcat(output700_800, "\\list700-800.txt");
        strcpy(output800_900, output);
        strcat(output800_900, "\\list800-900.txt");
        strcpy(output900_1000, output);
        strcat(output900_1000, "\\list900-1000.txt");
        strcpy(output1000_1100, output);
        strcat(output1000_1100, "\\list1000-1100.txt");
        strcpy(output1100_1200, output);
        strcat(output1100_1200, "\\list1100-1200.txt");
        strcpy(output1200_1300, output);
        strcat(output1200_1300, "\\list1200-1300.txt");
        strcpy(output1300_1400, output);
        strcat(output1300_1400, "\\list1300-1400.txt");
        strcpy(output1400_1500, output);
        strcat(output1400_1500, "\\list1400-1500.txt");
        strcpy(output1500_1600, output);
        strcat(output1500_1600, "\\list1500-1600.txt");
        strcpy(output1600_1700, output);
        strcat(output1600_1700, "\\list1600-1700.txt");
        strcpy(output1700_1800, output);
        strcat(output1700_1800, "\\list1700-1800.txt");
        strcpy(output1800_1900, output);
        strcat(output1800_1900, "\\list1800-1900.txt");
        strcpy(output1900_2000, output);
        strcat(output1900_2000, "\\list1900-2000.txt");
        strcpy(output2000_2100, output);
        strcat(output2000_2100, "\\list2000-2100.txt");
        strcpy(output2100_2200, output);
        strcat(output2100_2200, "\\list2100-2200.txt");
        strcpy(output2200_2300, output);
        strcat(output2200_2300, "\\list2200-2300.txt");
        strcpy(output2300_2400, output);
        strcat(output2300_2400, "\\list2300-2400.txt");
        strcpy(output2400_2500, output);
        strcat(output2400_2500, "\\list2400-2500.txt");
        strcpy(output2500_, output);
        strcat(output2500_, "\\list2500-.txt");
        results0.open(output0_100, ios::app);

        results1.open(output100_200, ios::app);

        results2.open(output200_300, ios::app);

        results3.open(output300_400, ios::app);

        results4.open(output400_500, ios::app);

        results5.open(output500_600, ios::app);

        results6.open(output600_700, ios::app);

        results7.open(output700_800, ios::app);

        results8.open(output800_900, ios::app);

        results9.open(output900_1000, ios::app);

        results10.open(output1000_1100, ios::app);

        results11.open(output1100_1200, ios::app);

        results12.open(output1200_1300, ios::app);

        results13.open(output1300_1400, ios::app);

        results14.open(output1400_1500, ios::app);

        results15.open(output1500_1600, ios::app);

        results16.open(output1600_1700, ios::app);

        results17.open(output1700_1800, ios::app);

        results18.open(output1800_1900, ios::app);

        results19.open(output1900_2000, ios::app);

        results20.open(output2000_2100, ios::app);

        results21.open(output2100_2200, ios::app);

        results22.open(output2200_2300, ios::app);

        results23.open(output2300_2400, ios::app);

        results24.open(output2400_2500, ios::app);

        results25.open(output2500_, ios::app);

        cout << "searchPath:" << searchPath << endl;
        cout << "searchDepth:" << searchDepth << endl;
        cout << "ignorePath:" << ignorePath << endl;
        cout << "output:" << output << endl;
        cout << "isIgnoreSearchDepth:" << isIgnoreSearchDepth << endl;
        char *p = NULL;
        char *_ignorepath = ignorePath;
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
        results0.close();
        results1.close();
        results2.close();
        results3.close();
        results4.close();
        results5.close();
        results6.close();
        results7.close();
        results8.close();
        results9.close();
        results10.close();
        results11.close();
        results12.close();
        results13.close();
        results14.close();
        results15.close();
        results16.close();
        results17.close();
        results18.close();
        results19.close();
        results20.close();
        results21.close();
        results22.close();
        results23.close();
        results24.close();
        results25.close();
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
    std::locale::global(std::locale(""));
    setSearchDepth(6);
    results0.open("D:\\Code\\C++\\output\\list0-100.txt", ios::app);
    searchFiles("D:\\Code\\C++\\TEST", "*");
    results0.close();
    getchar();
}

#endif