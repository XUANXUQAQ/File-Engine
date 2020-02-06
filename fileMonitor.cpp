#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <ctype.h>
//#define TEST

extern "C" __declspec(dllexport) void* fileWatcher(char path[300], char output[300], char closeSignalPosition[300]);

using namespace std;

__declspec(dllexport) void* fileWatcher(char path[300], char output[300], char closeSignalPosition[300]);
bool isMainExit(const char* close);

bool isMainExit(const char close[300]){
    fstream _file;
    _file.open(close, ios::in);
    if(!_file){
        return false;
    }else{
        return true;
    }
}


__declspec(dllexport) void* fileWatcher(char path[300], char output[300], char closeSignalPosition[300])
{
    int count = 0;
    DWORD cbBytes;
    char file_name[MAX_PATH]; //�����ļ���
    char file_rename[MAX_PATH]; //�����ļ��������������;
    char notify[1024];
    TCHAR* dir = (TCHAR*) path;
    ofstream file;
    char tip[] = "Start Monitor...\n";
    file.open(output);
    file << tip;
    file.close();
    


    HANDLE dirHandle = CreateFile(dir,
        GENERIC_READ | GENERIC_WRITE | FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        NULL);

    if (dirHandle == INVALID_HANDLE_VALUE) //�������ض����Ŀ���ļ�ϵͳ��֧�ָò���������ʧ�ܣ�ͬʱ����GetLastError()����ERROR_INVALID_FUNCTION
    {
        cout << "error" + GetLastError() << endl;
    }

    memset(file_name, 0, strlen(file_name));
    memset(file_rename, 0, strlen(file_rename));
    memset(notify, 0, strlen(notify));
    FILE_NOTIFY_INFORMATION *pnotify = (FILE_NOTIFY_INFORMATION*)notify;

    cout << tip;

    while (!isMainExit(closeSignalPosition))
    {
        if (ReadDirectoryChangesW(dirHandle, &notify, 1024, true,
            FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_SIZE,
            &cbBytes, NULL, NULL))
        {
            //ת���ļ���Ϊ���ֽ��ַ���;
            if (pnotify->FileName)
            {
                memset(file_name, 0, strlen(file_name));
                WideCharToMultiByte(CP_ACP, 0, pnotify->FileName, pnotify->FileNameLength / 2, file_name, 99, NULL, NULL);
            }

            //��ȡ���������ļ���;
            if (pnotify->NextEntryOffset != 0 && (pnotify->FileNameLength > 0 && pnotify->FileNameLength < MAX_PATH))
            {
                PFILE_NOTIFY_INFORMATION p = (PFILE_NOTIFY_INFORMATION)((char*)pnotify + pnotify->NextEntryOffset);
                memset(file_rename, 0, sizeof(file_rename));
                WideCharToMultiByte(CP_ACP, 0, p->FileName, p->FileNameLength / 2, file_rename, 99, NULL, NULL);
            }
            

            //�������͹�����,�����ļ����������ġ�ɾ������������;
            switch (pnotify->Action)
            {
                case FILE_ACTION_ADDED:
                    if (strstr(file_name, "$RECYCLE.BIN")==NULL){
                        cout << setw(5) << "file add : " << setw(5) << file_name << endl;
                        string data = "file add : ";
                        data.append(path);
                        data.append(file_name);
                        data.append("\n");                     
                        // ��׷��ģʽ���ļ�
                        count++;
                        if (count < 500){
                            ofstream outfile;
                            outfile.open(output, ios::app);
                            outfile << data;                    
                            outfile.close();
                        }else{
                            count = 0;
                            ofstream outfile;
                            outfile.open(output);
                            outfile << data;                    
                            outfile.close();
                        }
                    }
                    break;

                
                case FILE_ACTION_MODIFIED:
                    cout << "file modified : " << setw(5) << file_name << endl;
                    break;
                

                case FILE_ACTION_REMOVED:
                    if (strstr(file_name, "$RECYCLE.BIN")==NULL){
                        cout << setw(5) << "file removed : " << setw(5) << file_name << endl;
                        string data = "file removed : ";
                        data.append(path);
                        data.append(file_name);
                        data.append("\n");  data.append("\n");  
                        // ��׷��ģʽ���ļ�
                        count++;
                        if (count < 500){
                            ofstream outfile;
                            outfile.open(output, ios::app);
                            outfile << data;                    
                            outfile.close();
                        }else{
                            count = 0;
                            ofstream outfile;
                            outfile.open(output);
                            outfile << data;                    
                            outfile.close();
                        }
                    }
                    break;

                case FILE_ACTION_RENAMED_OLD_NAME:
                    if (strstr(file_name, "$RECYCLE.BIN")==NULL){
                        cout << "file renamed : " << setw(5) << file_name << "->" << file_rename << endl;
                        string data = "file renamed : ";
                        data.append(path);                    
                        data.append(file_name);
                        data.append("->");
                        data.append(path);
                        data.append(file_rename);
                        data.append("\n");  
                        // ��׷��ģʽ���ļ�
                        count++;
                        if (count < 500){
                            ofstream outfile;
                            outfile.open(output, ios::app);
                            outfile << data;                    
                            outfile.close();
                        }else{
                            count = 0;
                            ofstream outfile;
                            outfile.open(output);
                            outfile << data;                    
                            outfile.close();
                        }
                    }
                    break;

                default:
                    cout << "UNknow command!" << endl;
            }
        }
    }
    CloseHandle(dirHandle);
    char exit[] = "dll exit";
    cout << exit << endl;
    return 0;
}

#ifdef TEST
int main(int argc, char *argv[]){    
    char monitorPath[]= "D:\\";
    char outPut[] = "D:\\Code\\C++\\out.txt";
    char CLOSE[] = "D:\\Code\\C++\\CLOSE";
    fileWatcher(monitorPath, outPut, CLOSE);
    return 0;
}
#endif