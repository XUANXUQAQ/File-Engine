#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <ctype.h>
//#define TEST


extern "C" __declspec(dllexport) void monitor(char* path, char* output, char* closePosition);

using namespace std;

wchar_t fileName[300];
wchar_t fileRename[300];



int CountLines(char *filename)
{
    ifstream ReadFile;
    int n=0;
    string tmp;
    ReadFile.open(filename,ios::in);//ios::in 表示以只读的方式读取文件
    if(ReadFile.fail())//文件打开失败:返回0
    {
        return 0;
    }
    else//文件存在
    {
        while(getline(ReadFile,tmp,'\n'))
        {
            n++;
        }
        ReadFile.close();
        return n;
    }
}

bool isExist(const char* FileName)
{
    char FILENAME[600];
    strcpy(FILENAME, FileName);
    fstream _file;
    _file.open(FILENAME, ios::in);
    if(!_file)
    {
        return false;
    }
    else
    {
        return true;
    }
}

__declspec(dllexport) void monitor(char* path, char* output, char* closePosition){
    DWORD cbBytes;
    char file_name[MAX_PATH]; //设置文件名
    char file_rename[MAX_PATH]; //设置文件重命名后的名字;
    char notify[1024];
    char _path[300];
    ofstream file;
    strcpy(_path, path);
    cout << "Start Monitor..." << _path << endl;
    char OUTPUT[300];
    char CLOSE[300];
    strcpy(OUTPUT, output);
    strcpy(CLOSE, closePosition);
    char fileRemoved[300];
    memset(fileRemoved, 0, 300);
    char fileAdded[300];
    memset(fileAdded, 0 , 300);
    strcpy(fileRemoved, OUTPUT);
    strcat(fileRemoved, "\\fileRemoved.txt");
    strcpy(fileAdded, OUTPUT);
    strcat(fileAdded, "\\fileAdded.txt");
 
    FILE_NOTIFY_INFORMATION *pnotify = (FILE_NOTIFY_INFORMATION*)notify;
    TCHAR* dir = (TCHAR*) _path;
    HANDLE dirHandle = CreateFile(dir,
        GENERIC_READ | GENERIC_WRITE | FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        NULL);

    if (dirHandle == INVALID_HANDLE_VALUE) //若网络重定向或目标文件系统不支持该操作，函数失败，同时调用GetLastError()返回ERROR_INVALID_FUNCTION
    {
        cout << "error" + GetLastError() << endl;
    }

    while (!isExist(closePosition)){
        if (ReadDirectoryChangesW(dirHandle, &notify, 1024, true,
                FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_SIZE,
                &cbBytes, NULL, NULL))
            {
                //转换文件名为多字节字符串;
                if (pnotify->FileName)
                {
                    memset(file_name, 0, sizeof(file_name));
                    memset(fileName, 0, sizeof(fileName));               
                    wcscpy(fileName, pnotify->FileName);
                    WideCharToMultiByte(CP_ACP, 0, pnotify->FileName, pnotify->FileNameLength / 2, file_name, 250, NULL, NULL);
                }

                //获取重命名的文件名;
                if (pnotify->NextEntryOffset != 0 && (pnotify->FileNameLength > 0 && pnotify->FileNameLength < MAX_PATH))
                {
                    PFILE_NOTIFY_INFORMATION p = (PFILE_NOTIFY_INFORMATION)((char*)pnotify + pnotify->NextEntryOffset);
                    memset(file_rename, 0, sizeof(file_rename));
                    memset(fileRename, 0, sizeof(fileRename));
                    wcscpy(fileRename, pnotify->FileName);
                    WideCharToMultiByte(CP_ACP, 0, p->FileName, p->FileNameLength / 2, file_rename, 250, NULL, NULL);
                }

                if (file_name[strlen(file_name)-1] == '~'){
                    file_name[strlen(file_name)-1] = '\0';
                }
                

                //设置类型过滤器,监听文件创建、更改、删除、重命名等;
                switch (pnotify->Action)
                {
                    case FILE_ACTION_ADDED:
                        if (strstr(file_name, "$RECYCLE.BIN")==NULL){                            
                            //cout << "file add : " << file_name << endl; 
                            cout << "file add : ";       
                            string data;
                            data.append(path);
                            data.append(file_name);                     
                            cout << data << endl;
							ofstream outfile;
                            outfile.open(fileAdded, ios::app);
                            outfile << data << endl;
                            outfile.close();
                        }
                        break;

                    
                    case FILE_ACTION_MODIFIED:
                        if (strstr(file_name, "$RECYCLE.BIN")==NULL && strstr(file_name, "fileAdded.txt") == NULL && strstr(file_name, "fileRemoved.txt") == NULL){                            
                            //cout << "file add : " << file_name << endl; 
                            cout << "file add : ";       
                            string data;
                            data.append(path);
                            data.append(file_name);                     
                            cout << data << endl;
							ofstream outfile;
                            outfile.open(fileAdded, ios::app);
                            outfile << data << endl;
                            outfile.close();
                        }
                        break;
                    

                    case FILE_ACTION_REMOVED:
                        if (strstr(file_name, "$RECYCLE.BIN")==NULL){
                            //cout << setw(5) << "file removed : " << path << setw(5) << file_name << endl;
                            cout << "file removed : ";                      
                            string data;
                            data.append(path);
                            data.append(file_name);
                            cout << data << endl;   
							ofstream outfile;  

                            outfile.open(fileRemoved, ios::app);
                            outfile << data <<endl;
                            outfile.close();
                        
                        }
                        break;

                    case FILE_ACTION_RENAMED_OLD_NAME:
                        if (strstr(file_name, "$RECYCLE.BIN")==NULL){
                            //cout << "file renamed : " << setw(5) << path << file_name << "->" << path << file_rename << endl;  

                            cout << "file renamed : ";        
                            string data;
                            data.append(path);                    
                            data.append(file_name);
                            cout << data << "->";
                            ofstream outfile;  
                            outfile.open(fileRemoved, ios::app);
                            outfile << data <<endl;
                            outfile.close();
                        
                            data.clear();
                            data.append(path);
                            data.append(file_rename);   
                            cout << data << endl;     

                            outfile.open(fileAdded, ios::app);
                            outfile << data << endl;
                            outfile.close();
                                                                        
                        }
                        break;

                    default:
                        cout << "Unknown command!" << endl;
                }
            }
    }
    CloseHandle(dirHandle);
    cout << "exit" << endl;
}


#ifdef TEST
int main(){    
    char monitorPath[]= "C:\\Users\\13927\\Desktop\\test";
    monitor(monitorPath, "D:\\Code\\C++", "D:\\Code\\C++\\CLOSE");
    return 0;
}
#endif