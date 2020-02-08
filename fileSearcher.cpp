#include <string>  
#include <io.h>  
#include <iostream>  
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;  
//#define TEST


extern "C" __declspec(dllexport) void addIgnorePath(const char* path);
extern "C" __declspec(dllexport) void searchFiles(const char* path, const char* exd);
extern "C" __declspec(dllexport) void setSearchDepth(int i);
extern "C" __declspec(dllexport) char* getResult();
extern "C" __declspec(dllexport) void deleteResult();
extern "C" __declspec(dllexport) bool ResultReady();

vector<string> ignorePath;
vector<string> files;  
int searchDepth;
string results;
char* cstr;
bool isResultReady = false;

bool ResultReady();
void searchFiles(const char* path, const char* exd);
char* getResult();
void deleteResult();
void addIgnorePath(const char* path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);

__declspec(dllexport) bool ResultReady(){
    return isResultReady;
}

__declspec(dllexport) void searchFiles(const char* path, const char* exd){
    string file(path);
    string suffix(exd);
    isResultReady = false;
    results.clear();
    files.clear();
    cout << "start Search" << endl;
    search(file, exd);
    cout << "end Search" << endl;
    isResultReady = true;
    //cout << isResultReady << endl;
}

__declspec(dllexport) char* getResult(){
    int size = files.size();  
    for (int i = 0; i < size; i++)  
    { 
        results.append(files[i]);
        results.append("\n");
    } 
    cstr = new char[results.length() +1];
    strcpy(cstr, results.c_str());
    return cstr;
}

__declspec(dllexport) void deleteResult(){
    delete[] cstr;
}

__declspec(dllexport) void addIgnorePath(const char* path){
    string str(path);
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    ignorePath.push_back(str);
}

__declspec(dllexport) void setSearchDepth(int i){
    searchDepth = i;
}

int count(string path, string pattern){
    int begin = -1;
    int count = 0;
    while((begin=path.find(pattern,begin+1))!=string::npos)
    {
	    count++;
        begin=begin+pattern.length();
    }
    return count;
}

bool isSearchDepthOut(string path){
    int num = count(path, "\\");
    if (num > searchDepth-2){
        return true;
    }
    return false;
}

bool isIgnore(string path){
    if (path.find("$")!=string::npos){
        return true;
    }
    transform(path.begin(), path.end(), path.begin(), ::tolower);
    int size = ignorePath.size();
    for (int i = 0; i< size; i++){
        if (path.find(ignorePath[i]) != string::npos){
            return true;
        }
    }
    return false;
}

 
void search(string path, string exd)  
{  
    //cout << "getFiles()" << path<< endl;   
    //文件句柄  
    long   hFile = 0;  
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
            if ((fileinfo.attrib &  _A_SUBDIR))  
            {  
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
                    files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));  
                    bool SearchDepthOut = isSearchDepthOut(path);
                    bool Ignore = isIgnore(path);
                    bool result = !Ignore && !SearchDepthOut;
                    if (result){
                        search(pathName.assign(path).append("\\").append(fileinfo.name), exd); 
                        //cout << isResultReady << endl;
                    } 
                }
            }  
            else  
            {  
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)  
                    files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));  
            }  
        } while (_findnext(hFile, &fileinfo) == 0);  
        _findclose(hFile);  
    }
}  
  
#ifdef TEST
int main()  
{  
    char *result;
    addIgnorePath("C:\\Windows");
    addIgnorePath("C:\\Users");
    setSearchDepth(6);
    searchFiles("D:", "*");  
    result = getResult();
    cout << result << endl;
    deleteResult();
    system("pause");
    return 0;
}  
#endif