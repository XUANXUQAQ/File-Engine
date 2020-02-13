#include <string>  
#include <io.h>  
#include <iostream>  
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
using namespace std;  



vector<string> ignorePath;
int searchDepth;
char* searchPath;
string results;


void searchFiles(const char* path, const char* exd);
void addIgnorePath(const char* path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);
void clearResults();

void searchFilesIgnoreSearchDepth(const char* path, const char* exd){
    string file(path);
    string suffix(exd);
    cout << "start search without searchDepth" << endl;
    searchIgnoreSearchDepth(file, suffix);
    cout <<"end search without searchDepth" <<endl;
}


void searchIgnoreSearchDepth(string path, string exd){
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
                    results.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    results.append("\n");
                    bool Ignore = isIgnore(path);
                    if (!Ignore){
                        searchIgnoreSearchDepth(pathName.assign(path).append("\\").append(fileinfo.name), exd); 
                        //cout << isResultReady << endl;
                    } 
                }
            }  
            else  
            {  
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)  
                    results.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    results.append("\n");
            }  
        } while (_findnext(hFile, &fileinfo) == 0);  
        _findclose(hFile);  
    }
}


void searchFiles(const char* path, const char* exd){
    string file(path);
    string suffix(exd);
    cout << "start Search" << endl;
    search(file, exd);
    cout << "end Search" << endl;
    //cout << isResultReady << endl;
}


void addIgnorePath(const char* path){
    string str(path);
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    ignorePath.push_back(str);
}

void setSearchDepth(int i){
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
                    results.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    results.append("\n");

                    bool SearchDepthOut = isSearchDepthOut(path);
                    bool Ignore = isIgnore(path);
                    bool result = !Ignore && !SearchDepthOut;
                    if (result){
                        search(pathName.assign(path).append("\\").append(fileinfo.name), exd);                    
                    } 
                }
            }  
            else  
            {  
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)  
                    results.append(pathName.assign(path).append("\\").append(fileinfo.name));
                    results.append("\n");
            }  
        } while (_findnext(hFile, &fileinfo) == 0);  
        _findclose(hFile);  
    }
}  
  

int main(int argc, char* argv[])  
{  
    char searchPath[260];
    char searchDepth[50];
    char ignorePath[3000];
    char output[260];
    char isIgnoreSearchDepth[2];
    ofstream outfile;
    if (argc == 6){
        strcpy(searchPath, argv[1]);
        strcpy(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy(ignorePath, argv[3]);
        strcpy(output, argv[4]);
        strcpy(isIgnoreSearchDepth, argv[5]);
        cout << "searchPath:" << searchPath <<endl;
        cout << "searchDepth:" << searchDepth <<endl;
        cout << "ignorePath:" << ignorePath <<endl;
        cout << "output:" << output <<endl;
        cout << "isIgnoreSearchDepth:" << isIgnoreSearchDepth << endl; 
        char *p = NULL;
        char *_ignorepath = ignorePath;
        p = strtok(_ignorepath, ",");
        if (p != NULL){
            addIgnorePath(p);
            cout << "adding ignorePath:" << p << endl;
        }
        
        while (p != NULL){
            p = strtok(NULL, ",");
            if (p != NULL){
                addIgnorePath(p);
                cout << "adding ignorePath:" << p << endl;
            }
        }
        clock_t startTime,endTime;
        startTime = clock();
        if (atoi(isIgnoreSearchDepth) == 1){
            cout << "ignore searchDepth!!!" << endl;
            searchFilesIgnoreSearchDepth(searchPath, "*");
            //将结果写入文件                    
            outfile.open(output);               
            outfile << results << endl;
            outfile.close();
        }else{
            cout << "normal search" << endl;
            searchFiles(searchPath, "*");  
            //将结果写入文件                    
            outfile.open(output);               
            outfile << results << endl;
            outfile.close();
        }
        endTime = clock();
        cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
        return 0;
    }else{
        cout << "args error" << endl;
    } 
}  
