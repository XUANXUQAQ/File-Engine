#include <stdio.h>
#include <windows.h>
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) // ������ڵ�ַ

int main(){
    int hwndDOS = GetForegroundWindow();
    ShowWindow(hwndDOS, SW_HIDE);
    char *path = "fileToOpen.txt";
    char command[260];
    char workingdir[260];
    FILE *fp;
    fp = fopen(path, "r");
    //��ȡ�ļ��еĿ�ִ���ļ���ַ
    fgets(command, 260, fp);
    command[strlen(command)-1]='\0';//ȥ�����з�
    //��ȡ�����ļ���
    fgets(workingdir, 260, fp);
    fclose(fp);
    ShellExecute(NULL,"open",command,NULL,workingdir,SW_SHOWNORMAL);
}