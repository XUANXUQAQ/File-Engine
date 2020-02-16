#include <string>
#include <io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
using namespace std;

vector<string> ignorePathVector;
int searchDepth;
char *searchPath;
string resultsA;
string resultsB;
string resultsC;
string resultsD;
string resultsE;
string resultsF;
string resultsG;
string resultsH;
string resultsI;
string resultsJ;
string resultsK;
string resultsL;
string resultsM;
string resultsN;
string resultsO;
string resultsP;
string resultsQ;
string resultsR;
string resultsS;
string resultsT;
string resultsU;
string resultsV;
string resultsW;
string resultsX;
string resultsY;
string resultsZ;
string resultsNum;
string resultsPercentSign;
string resultsUnderline;
string resultsUnique;

void searchFiles(const char *path, const char *exd);
void addIgnorePath(const char *path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);
void clearResults();

char toUpper(char i)
{
    if ((i >= 65) && (i <= 90))
    {
        return i;
    }
    else if ((i >= 97) && (i <= 122))
    {
        return i - 32;
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
                    switch (toUpper(fileinfo.name[0]))
                    {
                    case 'A':
                        resultsA.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsA.append("\n");
                        break;
                    case 'B':
                        resultsB.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsB.append("\n");
                        break;
                    case 'C':
                        resultsC.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsC.append("\n");
                        break;
                    case 'D':
                        resultsD.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsD.append("\n");
                        break;
                    case 'E':
                        resultsE.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsE.append("\n");
                        break;
                    case 'F':
                        resultsF.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsF.append("\n");
                        break;
                    case 'G':
                        resultsG.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsG.append("\n");
                        break;
                    case 'H':
                        resultsH.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsH.append("\n");
                        break;
                    case 'I':
                        resultsI.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsI.append("\n");
                        break;
                    case 'J':
                        resultsJ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsJ.append("\n");
                        break;
                    case 'L':
                        resultsL.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsL.append("\n");
                        break;
                    case 'M':
                        resultsM.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsM.append("\n");
                        break;
                    case 'N':
                        resultsN.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsN.append("\n");
                        break;
                    case 'O':
                        resultsO.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsO.append("\n");
                        break;
                    case 'P':
                        resultsP.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsP.append("\n");
                        break;
                    case 'Q':
                        resultsQ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsQ.append("\n");
                        break;
                    case 'R':
                        resultsR.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsR.append("\n");
                        break;
                    case 'S':
                        resultsS.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsS.append("\n");
                        break;
                    case 'T':
                        resultsT.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsT.append("\n");
                        break;
                    case 'U':
                        resultsU.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsU.append("\n");
                        break;
                    case 'V':
                        resultsV.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsV.append("\n");
                        break;
                    case 'W':
                        resultsW.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsW.append("\n");
                        break;
                    case 'X':
                        resultsX.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsX.append("\n");
                        break;
                    case 'Y':
                        resultsY.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsY.append("\n");
                        break;
                    case 'Z':
                        resultsZ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsZ.append("\n");
                        break;
                    case '%':
                        resultsPercentSign.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsPercentSign.append("\n");
                    case '_':
                        resultsUnderline.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsUnderline.append("\n");
                    default:
                        if (strstr(fileinfo.name, "Num") != NULL)
                        {
                            resultsNum.append(pathName.assign(path).append("\\").append(fileinfo.name));
                            resultsNum.append("\n");
                        }
                        else
                        {
                            resultsUnique.append(pathName.assign(path).append("\\").append(fileinfo.name));
                            resultsUnique.append("\n");
                        }
                        break;
                    }

                    bool Ignore = isIgnore(path);
                    if (!Ignore)
                    {
                        searchIgnoreSearchDepth(pathName.assign(path).append("\\").append(fileinfo.name), exd);
                        //cout << isResultReady << endl;
                    }
                }
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    switch (toUpper(fileinfo.name[0]))
                    {
                    case 'A':
                        resultsA.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsA.append("\n");
                        break;
                    case 'B':
                        resultsB.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsB.append("\n");
                        break;
                    case 'C':
                        resultsC.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsC.append("\n");
                        break;
                    case 'D':
                        resultsD.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsD.append("\n");
                        break;
                    case 'E':
                        resultsE.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsE.append("\n");
                        break;
                    case 'F':
                        resultsF.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsF.append("\n");
                        break;
                    case 'G':
                        resultsG.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsG.append("\n");
                        break;
                    case 'H':
                        resultsH.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsH.append("\n");
                        break;
                    case 'I':
                        resultsI.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsI.append("\n");
                        break;
                    case 'J':
                        resultsJ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsJ.append("\n");
                        break;
                    case 'L':
                        resultsL.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsL.append("\n");
                        break;
                    case 'M':
                        resultsM.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsM.append("\n");
                        break;
                    case 'N':
                        resultsN.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsN.append("\n");
                        break;
                    case 'O':
                        resultsO.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsO.append("\n");
                        break;
                    case 'P':
                        resultsP.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsP.append("\n");
                        break;
                    case 'Q':
                        resultsQ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsQ.append("\n");
                        break;
                    case 'R':
                        resultsR.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsR.append("\n");
                        break;
                    case 'S':
                        resultsS.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsS.append("\n");
                        break;
                    case 'T':
                        resultsT.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsT.append("\n");
                        break;
                    case 'U':
                        resultsU.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsU.append("\n");
                        break;
                    case 'V':
                        resultsV.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsV.append("\n");
                        break;
                    case 'W':
                        resultsW.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsW.append("\n");
                        break;
                    case 'X':
                        resultsX.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsX.append("\n");
                        break;
                    case 'Y':
                        resultsY.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsY.append("\n");
                        break;
                    case 'Z':
                        resultsZ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsZ.append("\n");
                        break;
                    case '%':
                        resultsPercentSign.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsPercentSign.append("\n");
                    case '_':
                        resultsUnderline.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsUnderline.append("\n");
                    default:
                        if (strstr(fileinfo.name, "Num") != NULL)
                        {
                            resultsNum.append(pathName.assign(path).append("\\").append(fileinfo.name));
                            resultsNum.append("\n");
                        }
                        else
                        {
                            resultsUnique.append(pathName.assign(path).append("\\").append(fileinfo.name));
                            resultsUnique.append("\n");
                        }
                        break;
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
    //cout << isResultReady << endl;
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

                    if (strstr(fileinfo.name, "A") != NULL)
                    {
                        resultsA.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsA.append("\n");
                    }
                    if (strstr(fileinfo.name, "B") != NULL)
                    {
                        resultsB.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsB.append("\n");
                    }
                    if (strstr(fileinfo.name, "C") != NULL)
                    {
                        resultsC.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsC.append("\n");
                    }
                    if (strstr(fileinfo.name, "D") != NULL)
                    {
                        resultsD.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsD.append("\n");
                    }
                    if (strstr(fileinfo.name, "E") != NULL)
                    {
                        resultsE.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsE.append("\n");
                    }
                    if (strstr(fileinfo.name, "F") != NULL)
                    {
                        resultsF.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsF.append("\n");
                    }
                    if (strstr(fileinfo.name, "G") != NULL)
                    {
                        resultsG.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsG.append("\n");
                    }
                    if (strstr(fileinfo.name, "H") != NULL)
                    {
                        resultsH.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsH.append("\n");
                    }
                    if (strstr(fileinfo.name, "I") != NULL)
                    {
                        resultsI.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsI.append("\n");
                    }
                    if (strstr(fileinfo.name, "J") != NULL)
                    {
                        resultsJ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsJ.append("\n");
                    }
                    if (strstr(fileinfo.name, "L") != NULL)
                    {
                        resultsL.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsL.append("\n");
                    }
                    if (strstr(fileinfo.name, "M") != NULL)
                    {
                        resultsM.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsM.append("\n");
                    }
                    if (strstr(fileinfo.name, "N") != NULL)
                    {
                        resultsN.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsN.append("\n");
                    }
                    if (strstr(fileinfo.name, "O") != NULL)
                    {
                        resultsO.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsO.append("\n");
                    }
                    if (strstr(fileinfo.name, "P") != NULL)
                    {
                        resultsP.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsP.append("\n");
                    }
                    if (strstr(fileinfo.name, "Q") != NULL)
                    {
                        resultsQ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsQ.append("\n");
                    }
                    if (strstr(fileinfo.name, "R") != NULL)
                    {
                        resultsR.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsR.append("\n");
                    }
                    if (strstr(fileinfo.name, "S") != NULL)
                    {
                        resultsS.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsS.append("\n");
                    }
                    if (strstr(fileinfo.name, "T") != NULL)
                    {
                        resultsT.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsT.append("\n");
                    }
                    if (strstr(fileinfo.name, "U") != NULL)
                    {
                        resultsU.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsU.append("\n");
                    }
                    if (strstr(fileinfo.name, "V") != NULL)
                    {
                        resultsV.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsV.append("\n");
                    }
                    if (strstr(fileinfo.name, "W") != NULL)
                    {
                        resultsW.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsW.append("\n");
                    }
                    if (strstr(fileinfo.name, "X") != NULL)
                    {
                        resultsX.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsX.append("\n");
                    }
                    if (strstr(fileinfo.name, "Y") != NULL)
                    {
                        resultsY.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsY.append("\n");
                    }
                    if (strstr(fileinfo.name, "Z") != NULL)
                    {
                        resultsZ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsZ.append("\n");
                    }
                    if (strstr(fileinfo.name, "%") != NULL)
                    {
                        resultsPercentSign.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsPercentSign.append("\n");
                    }
                    if (strstr(fileinfo.name, "_") != NULL)
                    {
                        resultsUnderline.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsUnderline.append("\n");
                    }
                    if (strstr(fileinfo.name, "Num") != NULL)
                    {
                        resultsNum.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsNum.append("\n");
                    }
                    else
                    {
                        resultsUnique.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsUnique.append("\n");
                    }

                    bool SearchDepthOut = isSearchDepthOut(path);
                    bool Ignore = isIgnore(path);
                    bool result = !Ignore && !SearchDepthOut;
                    if (result)
                    {
                        search(pathName.assign(path).append("\\").append(fileinfo.name), exd);
                    }
                }
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    if (strstr(fileinfo.name, "A") != NULL)
                    {
                        resultsA.append(pathName.assign(path).append("\\").append(fileinfo.name));
                        resultsA.append("\n");
                    }
                if (strstr(fileinfo.name, "B") != NULL)
                {
                    resultsB.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsB.append("\n");
                }
                if (strstr(fileinfo.name, "C") != NULL)
                {
                    resultsC.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsC.append("\n");
                }
                if (strstr(fileinfo.name, "D") != NULL)
                {
                    resultsD.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsD.append("\n");
                }
                if (strstr(fileinfo.name, "E") != NULL)
                {
                    resultsE.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsE.append("\n");
                }
                if (strstr(fileinfo.name, "F") != NULL)
                {
                    resultsF.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsF.append("\n");
                }
                if (strstr(fileinfo.name, "G") != NULL)
                {
                    resultsG.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsG.append("\n");
                }
                if (strstr(fileinfo.name, "H") != NULL)
                {
                    resultsH.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsH.append("\n");
                }
                if (strstr(fileinfo.name, "I") != NULL)
                {
                    resultsI.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsI.append("\n");
                }
                if (strstr(fileinfo.name, "J") != NULL)
                {
                    resultsJ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsJ.append("\n");
                }
                if (strstr(fileinfo.name, "L") != NULL)
                {
                    resultsL.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsL.append("\n");
                }
                if (strstr(fileinfo.name, "M") != NULL)
                {
                    resultsM.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsM.append("\n");
                }
                if (strstr(fileinfo.name, "N") != NULL)
                {
                    resultsN.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsN.append("\n");
                }
                if (strstr(fileinfo.name, "O") != NULL)
                {
                    resultsO.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsO.append("\n");
                }
                if (strstr(fileinfo.name, "P") != NULL)
                {
                    resultsP.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsP.append("\n");
                }
                if (strstr(fileinfo.name, "Q") != NULL)
                {
                    resultsQ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsQ.append("\n");
                }
                if (strstr(fileinfo.name, "R") != NULL)
                {
                    resultsR.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsR.append("\n");
                }
                if (strstr(fileinfo.name, "S") != NULL)
                {
                    resultsS.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsS.append("\n");
                }
                if (strstr(fileinfo.name, "T") != NULL)
                {
                    resultsT.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsT.append("\n");
                }
                if (strstr(fileinfo.name, "U") != NULL)
                {
                    resultsU.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsU.append("\n");
                }
                if (strstr(fileinfo.name, "V") != NULL)
                {
                    resultsV.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsV.append("\n");
                }
                if (strstr(fileinfo.name, "W") != NULL)
                {
                    resultsW.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsW.append("\n");
                }
                if (strstr(fileinfo.name, "X") != NULL)
                {
                    resultsX.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsX.append("\n");
                }
                if (strstr(fileinfo.name, "Y") != NULL)
                {
                    resultsY.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsY.append("\n");
                }
                if (strstr(fileinfo.name, "Z") != NULL)
                {
                    resultsZ.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsZ.append("\n");
                }
                if (strstr(fileinfo.name, "%") != NULL)
                {
                    resultsPercentSign.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsPercentSign.append("\n");
                }
                if (strstr(fileinfo.name, "_") != NULL)
                {
                    resultsUnderline.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsUnderline.append("\n");
                }
                if (strstr(fileinfo.name, "Num") != NULL)
                {
                    resultsNum.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsNum.append("\n");
                }
                else
                {
                    resultsUnique.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    resultsUnique.append("\n");
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

int main(int argc, char *argv[])
{
    char searchPath[260];
    char searchDepth[50];
    char ignorePath[3000];
    char output[260];
    char isIgnoreSearchDepth[2];
    char outputA[260];
    char outputB[260];
    char outputC[260];
    char outputD[260];
    char outputE[260];
    char outputF[260];
    char outputG[260];
    char outputH[260];
    char outputI[260];
    char outputJ[260];
    char outputK[260];
    char outputL[260];
    char outputM[260];
    char outputN[260];
    char outputO[260];
    char outputP[260];
    char outputQ[260];
    char outputR[260];
    char outputS[260];
    char outputT[260];
    char outputU[260];
    char outputV[260];
    char outputW[260];
    char outputX[260];
    char outputY[260];
    char outputZ[260];
    char outputNum[260];
    char outputPercentSign[260];
    char outputUnderline[260];
    char outputUnique[260];

    ofstream outfile;
    if (argc == 6)
    {
        strcpy(searchPath, argv[1]);
        strcpy(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy(ignorePath, argv[3]);
        strcpy(output, argv[4]);
        strcpy(isIgnoreSearchDepth, argv[5]);
        strcpy(outputA, output);
        strcat(outputA, "\\listA.txt");
        strcpy(outputB, output);
        strcat(outputB, "\\listB.txt");
        strcpy(outputC, output);
        strcat(outputC, "\\listC.txt");
        strcpy(outputD, output);
        strcat(outputD, "\\listD.txt");
        strcpy(outputE, output);
        strcat(outputE, "\\listE.txt");
        strcpy(outputF, output);
        strcat(outputF, "\\listF.txt");
        strcpy(outputG, output);
        strcat(outputG, "\\listG.txt");
        strcpy(outputH, output);
        strcat(outputH, "\\listH.txt");
        strcpy(outputI, output);
        strcat(outputI, "\\listI.txt");
        strcpy(outputJ, output);
        strcat(outputJ, "\\listJ.txt");
        strcpy(outputK, output);
        strcat(outputK, "\\listK.txt");
        strcpy(outputL, output);
        strcat(outputL, "\\listL.txt");
        strcpy(outputM, output);
        strcat(outputM, "\\listM.txt");
        strcpy(outputN, output);
        strcat(outputN, "\\listN.txt");
        strcpy(outputO, output);
        strcat(outputO, "\\listO.txt");
        strcpy(outputP, output);
        strcat(outputP, "\\listP.txt");
        strcpy(outputQ, output);
        strcat(outputQ, "\\listQ.txt");
        strcpy(outputR, output);
        strcat(outputR, "\\listR.txt");
        strcpy(outputS, output);
        strcat(outputS, "\\listS.txt");
        strcpy(outputT, output);
        strcat(outputT, "\\listT.txt");
        strcpy(outputU, output);
        strcat(outputU, "\\listU.txt");
        strcpy(outputV, output);
        strcat(outputV, "\\listV.txt");
        strcpy(outputW, output);
        strcat(outputW, "\\listW.txt");
        strcpy(outputX, output);
        strcat(outputX, "\\listX.txt");
        strcpy(outputY, output);
        strcat(outputY, "\\listY.txt");
        strcpy(outputZ, output);
        strcat(outputZ, "\\listZ.txt");
        strcpy(outputNum, output);
        strcat(outputNum, "\\listNum.txt");
        strcpy(outputPercentSign, output);
        strcat(outputPercentSign, "\\listPercentSign.txt");
        strcpy(outputUnderline, output);
        strcat(outputUnderline, "\\listUnderline.txt");
        strcpy(outputUnique, output);
        strcat(outputUnique, "\\listUnique.txt");
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
            outfile.open(outputA, ios::app);
            outfile << resultsA << endl;
            outfile.close();
            outfile.open(outputB, ios::app);
            outfile << resultsB << endl;
            outfile.close();
            outfile.open(outputC, ios::app);
            outfile << resultsC << endl;
            outfile.close();
            outfile.open(outputD, ios::app);
            outfile << resultsD << endl;
            outfile.close();
            outfile.open(outputE, ios::app);
            outfile << resultsE << endl;
            outfile.close();
            outfile.open(outputF, ios::app);
            outfile << resultsF << endl;
            outfile.close();
            outfile.open(outputG, ios::app);
            outfile << resultsG << endl;
            outfile.close();
            outfile.open(outputH, ios::app);
            outfile << resultsH << endl;
            outfile.close();
            outfile.open(outputI, ios::app);
            outfile << resultsI << endl;
            outfile.close();
            outfile.open(outputJ, ios::app);
            outfile << resultsJ << endl;
            outfile.close();
            outfile.open(outputK, ios::app);
            outfile << resultsK << endl;
            outfile.close();
            outfile.open(outputL, ios::app);
            outfile << resultsL << endl;
            outfile.close();
            outfile.open(outputM, ios::app);
            outfile << resultsM << endl;
            outfile.close();
            outfile.open(outputN, ios::app);
            outfile << resultsN << endl;
            outfile.close();
            outfile.open(outputO, ios::app);
            outfile << resultsO << endl;
            outfile.close();
            outfile.open(outputP, ios::app);
            outfile << resultsP << endl;
            outfile.close();
            outfile.open(outputQ, ios::app);
            outfile << resultsQ << endl;
            outfile.close();
            outfile.open(outputR, ios::app);
            outfile << resultsR << endl;
            outfile.close();
            outfile.open(outputS, ios::app);
            outfile << resultsS << endl;
            outfile.close();
            outfile.open(outputT, ios::app);
            outfile << resultsT << endl;
            outfile.close();
            outfile.open(outputU, ios::app);
            outfile << resultsU << endl;
            outfile.close();
            outfile.open(outputV, ios::app);
            outfile << resultsV << endl;
            outfile.close();
            outfile.open(outputW, ios::app);
            outfile << resultsW << endl;
            outfile.close();
            outfile.open(outputX, ios::app);
            outfile << resultsX << endl;
            outfile.close();
            outfile.open(outputY, ios::app);
            outfile << resultsY << endl;
            outfile.close();
            outfile.open(outputZ, ios::app);
            outfile << resultsZ << endl;
            outfile.close();
            outfile.open(outputNum, ios::app);
            outfile << resultsNum << endl;
            outfile.close();
            outfile.open(outputPercentSign, ios::app);
            outfile << resultsPercentSign << endl;
            outfile.close();
            outfile.open(outputUnderline, ios::app);
            outfile << resultsUnderline << endl;
            outfile.close();
            outfile.open(outputUnique, ios::app);
            outfile << resultsUnique << endl;
            outfile.close();
        }
        else
        {
            cout << "normal search" << endl;
            searchFiles(searchPath, "*");
            //将结果写入文件
            outfile.open(outputA, ios::app);
            outfile << resultsA << endl;
            outfile.close();
            outfile.open(outputB, ios::app);
            outfile << resultsB << endl;
            outfile.close();
            outfile.open(outputC, ios::app);
            outfile << resultsC << endl;
            outfile.close();
            outfile.open(outputD, ios::app);
            outfile << resultsD << endl;
            outfile.close();
            outfile.open(outputE, ios::app);
            outfile << resultsE << endl;
            outfile.close();
            outfile.open(outputF, ios::app);
            outfile << resultsF << endl;
            outfile.close();
            outfile.open(outputG, ios::app);
            outfile << resultsG << endl;
            outfile.close();
            outfile.open(outputH, ios::app);
            outfile << resultsH << endl;
            outfile.close();
            outfile.open(outputI, ios::app);
            outfile << resultsI << endl;
            outfile.close();
            outfile.open(outputJ, ios::app);
            outfile << resultsJ << endl;
            outfile.close();
            outfile.open(outputK, ios::app);
            outfile << resultsK << endl;
            outfile.close();
            outfile.open(outputL, ios::app);
            outfile << resultsL << endl;
            outfile.close();
            outfile.open(outputM, ios::app);
            outfile << resultsM << endl;
            outfile.close();
            outfile.open(outputN, ios::app);
            outfile << resultsN << endl;
            outfile.close();
            outfile.open(outputO, ios::app);
            outfile << resultsO << endl;
            outfile.close();
            outfile.open(outputP, ios::app);
            outfile << resultsP << endl;
            outfile.close();
            outfile.open(outputQ, ios::app);
            outfile << resultsQ << endl;
            outfile.close();
            outfile.open(outputR, ios::app);
            outfile << resultsR << endl;
            outfile.close();
            outfile.open(outputS, ios::app);
            outfile << resultsS << endl;
            outfile.close();
            outfile.open(outputT, ios::app);
            outfile << resultsT << endl;
            outfile.close();
            outfile.open(outputU, ios::app);
            outfile << resultsU << endl;
            outfile.close();
            outfile.open(outputV, ios::app);
            outfile << resultsV << endl;
            outfile.close();
            outfile.open(outputW, ios::app);
            outfile << resultsW << endl;
            outfile.close();
            outfile.open(outputX, ios::app);
            outfile << resultsX << endl;
            outfile.close();
            outfile.open(outputY, ios::app);
            outfile << resultsY << endl;
            outfile.close();
            outfile.open(outputZ, ios::app);
            outfile << resultsZ << endl;
            outfile.close();
            outfile.open(outputNum, ios::app);
            outfile << resultsNum << endl;
            outfile.close();
            outfile.open(outputPercentSign, ios::app);
            outfile << resultsPercentSign << endl;
            outfile.close();
            outfile.open(outputUnderline, ios::app);
            outfile << resultsUnderline << endl;
            outfile.close();
            outfile.open(outputUnique, ios::app);
            outfile << resultsUnique << endl;
            outfile.close();
        }

        return 0;
    }
    else
    {
        cout << "args error" << endl;
    }
}
