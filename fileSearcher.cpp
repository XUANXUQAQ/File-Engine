#include <string>
#include <io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <tchar.h>
#include <windows.h>

using namespace std;
//#define TEST

vector<string> ignorePathVector;
int searchDepth;
char *searchPath;
string results0_100;
string results100_200;
string results200_300;
string results300_400;
string results400_500;
string results500_600;
string results600_700;
string results700_800;
string results800_900;
string results900_1000;
string results1000_1100;
string results1100_1200;
string results1200_1300;
string results1300_1400;
string results1400_1500;
string results1500_1600;
string results1600_1700;
string results1700_1800;
string results1800_1900;
string results1900_2000;
string results2000_2100;
string results2100_2200;
string results2200_2300;
string results2300_2400;
string results2400_2500;
string results2500_;

void searchFiles(const char *path, const char *exd);
void addIgnorePath(const char *path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);
void clearResults();
bool includeChinese(string strs);

int getAscIISum(string name)
{
    if (includeChinese(name)){
        return 1;
    }
    int sum = 0;
    int length = name.length();
    for (int i = 0; i < length; i++)
    {
        sum += name[i];
    }
    return sum;
}

void saveResult(string path, int ascII)
{
    if (0 < ascII && ascII <= 100)
    {
        results0_100.append(path);
        results0_100.append("\n");
    }
    else if (100 < ascII && ascII <= 200)
    {
        results100_200.append(path);
        results100_200.append("\n");
    }
    else if (200 < ascII && ascII <= 300)
    {
        results200_300.append(path);
        results200_300.append("\n");
    }
    else if (300 < ascII && ascII <= 400)
    {
        results300_400.append(path);
        results300_400.append("\n");
    }
    else if (400 < ascII && ascII <= 500)
    {
        results400_500.append(path);
        results400_500.append("\n");
    }
    else if (500 < ascII && ascII <= 600)
    {
        results500_600.append(path);
        results500_600.append("\n");
    }
    else if (600 < ascII && ascII <= 700)
    {
        results600_700.append(path);
        results600_700.append("\n");
    }
    else if (700 < ascII && ascII <= 800)
    {
        results700_800.append(path);
        results700_800.append("\n");
    }
    else if (800 < ascII && ascII <= 900)
    {
        results800_900.append(path);
        results800_900.append("\n");
    }
    else if (900 < ascII && ascII <= 1000)
    {
        results900_1000.append(path);
        results900_1000.append("\n");
    }
    else if (1000 < ascII && ascII <= 1100)
    {
        results1000_1100.append(path);
        results1000_1100.append("\n");
    }
    else if (1100 < ascII && ascII <= 1200)
    {
        results1100_1200.append(path);
        results1100_1200.append("\n");
    }
    else if (1200 < ascII && ascII <= 1300)
    {
        results1200_1300.append(path);
        results1200_1300.append("\n");
    }
    else if (1300 < ascII && ascII <= 1400)
    {
        results1300_1400.append(path);
        results1300_1400.append("\n");
    }
    else if (1400 < ascII && ascII <= 1500)
    {
        results1400_1500.append(path);
        results1400_1500.append("\n");
    }
    else if (1500 < ascII && ascII <= 1600)
    {
        results1500_1600.append(path);
        results1500_1600.append("\n");
    }
    else if (1600 < ascII && ascII <= 1700)
    {
        results1600_1700.append(path);
        results1600_1700.append("\n");
    }
    else if (1700 < ascII && ascII <= 1800)
    {
        results1700_1800.append(path);
        results1700_1800.append("\n");
    }
    else if (1800 < ascII && ascII <= 1900)
    {
        results1800_1900.append(path);
        results1800_1900.append("\n");
    }
    else if (1900 < ascII && ascII <= 2000)
    {
        results1900_2000.append(path);
        results1900_2000.append("\n");
    }
    else if (2000 < ascII && ascII <= 2100)
    {
        results2000_2100.append(path);
        results2000_2100.append("\n");
    }
    else if (2100 < ascII && ascII <= 2200)
    {
        results2100_2200.append(path);
        results2100_2200.append("\n");
    }
    else if (2200 < ascII && ascII <= 2300)
    {
        results2200_2300.append(path);
        results2200_2300.append("\n");
    }
    else if (2300 < ascII && ascII <= 2400)
    {
        results2300_2400.append(path);
        results2300_2400.append("\n");
    }
    else if (2400 < ascII && ascII <= 2500)
    {
        results2400_2500.append(path);
        results2400_2500.append("\n");
    }
    else
    {
        results2500_.append(path);
        results2500_.append("\n");
    }
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

bool includeChinese(string strs)
{
    char _str[260];
    strcpy(_str, strs.c_str());
    char *str = _str;
    char c;
    while (1)
    {
        c = *str++;
        if (c == 0)
            break;    //如果到字符串尾则说明该字符串没有中文字符
        if (c & 0x80) //如果字符高位为1且下一字符高位也是1则有中文字符
            if (*str & 0x80)
                return true;
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
    HWND hwnd;  
    hwnd=FindWindow(TEXT("ConsoleWindowClass"),NULL); //处理顶级窗口的类名和窗口名称匹配指定的字符串,不搜索子窗口。  
    if(hwnd)  
    {  
        ShowWindow(hwnd,SW_HIDE);               //设置指定窗口的显示状态  
    }  
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

    ofstream outfile;
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
            //将结果写入文件
            outfile.open(output0_100, ios::app);
            outfile << results0_100 << endl;
            outfile.close();
            outfile.open(output100_200, ios::app);
            outfile << results100_200 << endl;
            outfile.close();
            outfile.open(output200_300, ios::app);
            outfile << results200_300 << endl;
            outfile.close();
            outfile.open(output300_400, ios::app);
            outfile << results300_400 << endl;
            outfile.close();
            outfile.open(output400_500, ios::app);
            outfile << results400_500 << endl;
            outfile.close();
            outfile.open(output500_600, ios::app);
            outfile << results500_600 << endl;
            outfile.close();
            outfile.open(output600_700, ios::app);
            outfile << results600_700 << endl;
            outfile.close();
            outfile.open(output700_800, ios::app);
            outfile << results700_800 << endl;
            outfile.close();
            outfile.open(output800_900, ios::app);
            outfile << results800_900 << endl;
            outfile.close();
            outfile.open(output900_1000, ios::app);
            outfile << results900_1000 << endl;
            outfile.close();
            outfile.open(output1000_1100, ios::app);
            outfile << results1000_1100 << endl;
            outfile.close();
            outfile.open(output1100_1200, ios::app);
            outfile << results1100_1200 << endl;
            outfile.close();
            outfile.open(output1200_1300, ios::app);
            outfile << results1200_1300 << endl;
            outfile.close();
            outfile.open(output1300_1400, ios::app);
            outfile << results1300_1400 << endl;
            outfile.close();
            outfile.open(output1400_1500, ios::app);
            outfile << results1400_1500 << endl;
            outfile.close();
            outfile.open(output1500_1600, ios::app);
            outfile << results1500_1600 << endl;
            outfile.close();
            outfile.open(output1600_1700, ios::app);
            outfile << results1600_1700 << endl;
            outfile.close();
            outfile.open(output1700_1800, ios::app);
            outfile << results1700_1800 << endl;
            outfile.close();
            outfile.open(output1800_1900, ios::app);
            outfile << results1800_1900 << endl;
            outfile.close();
            outfile.open(output1900_2000, ios::app);
            outfile << results1900_2000 << endl;
            outfile.close();
            outfile.open(output2000_2100, ios::app);
            outfile << results2000_2100 << endl;
            outfile.close();
            outfile.open(output2100_2200, ios::app);
            outfile << results2100_2200 << endl;
            outfile.close();
            outfile.open(output2200_2300, ios::app);
            outfile << results2200_2300 << endl;
            outfile.close();
            outfile.open(output2300_2400, ios::app);
            outfile << results2300_2400 << endl;
            outfile.close();
            outfile.open(output2400_2500, ios::app);
            outfile << results2400_2500 << endl;
            outfile.close();
            outfile.open(output2500_, ios::app);
            outfile << results2500_ << endl;
            outfile.close();
        }
        else
        {
            cout << "normal search" << endl;
            searchFiles(searchPath, "*");
            //将结果写入文件
            outfile.open(output0_100, ios::app);
            outfile << results0_100 << endl;
            outfile.close();
            outfile.open(output100_200, ios::app);
            outfile << results100_200 << endl;
            outfile.close();
            outfile.open(output200_300, ios::app);
            outfile << results200_300 << endl;
            outfile.close();
            outfile.open(output300_400, ios::app);
            outfile << results300_400 << endl;
            outfile.close();
            outfile.open(output400_500, ios::app);
            outfile << results400_500 << endl;
            outfile.close();
            outfile.open(output500_600, ios::app);
            outfile << results500_600 << endl;
            outfile.close();
            outfile.open(output600_700, ios::app);
            outfile << results600_700 << endl;
            outfile.close();
            outfile.open(output700_800, ios::app);
            outfile << results700_800 << endl;
            outfile.close();
            outfile.open(output800_900, ios::app);
            outfile << results800_900 << endl;
            outfile.close();
            outfile.open(output900_1000, ios::app);
            outfile << results900_1000 << endl;
            outfile.close();
            outfile.open(output1000_1100, ios::app);
            outfile << results1000_1100 << endl;
            outfile.close();
            outfile.open(output1100_1200, ios::app);
            outfile << results1100_1200 << endl;
            outfile.close();
            outfile.open(output1200_1300, ios::app);
            outfile << results1200_1300 << endl;
            outfile.close();
            outfile.open(output1300_1400, ios::app);
            outfile << results1300_1400 << endl;
            outfile.close();
            outfile.open(output1400_1500, ios::app);
            outfile << results1400_1500 << endl;
            outfile.close();
            outfile.open(output1500_1600, ios::app);
            outfile << results1500_1600 << endl;
            outfile.close();
            outfile.open(output1600_1700, ios::app);
            outfile << results1600_1700 << endl;
            outfile.close();
            outfile.open(output1700_1800, ios::app);
            outfile << results1700_1800 << endl;
            outfile.close();
            outfile.open(output1800_1900, ios::app);
            outfile << results1800_1900 << endl;
            outfile.close();
            outfile.open(output1900_2000, ios::app);
            outfile << results1900_2000 << endl;
            outfile.close();
            outfile.open(output2000_2100, ios::app);
            outfile << results2000_2100 << endl;
            outfile.close();
            outfile.open(output2100_2200, ios::app);
            outfile << results2100_2200 << endl;
            outfile.close();
            outfile.open(output2200_2300, ios::app);
            outfile << results2200_2300 << endl;
            outfile.close();
            outfile.open(output2300_2400, ios::app);
            outfile << results2300_2400 << endl;
            outfile.close();
            outfile.open(output2400_2500, ios::app);
            outfile << results2400_2500 << endl;
            outfile.close();
            outfile.open(output2500_, ios::app);
            outfile << results2500_ << endl;
            outfile.close();
        }

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
    setSearchDepth(6);
    searchFiles("D:", "*");
    getchar();
}

#endif