#include <stdio.h>
#include <windows.h>
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) // 设置入口地址

int main(){
    int hwndDOS = GetForegroundWindow();
    ShowWindow(hwndDOS, SW_HIDE);
    char *path = "fileToOpen.txt";
    char command[260];
    char workingdir[260];
    FILE *fp;
    fp = fopen(path, "r");
    //读取文件中的可执行文件地址
    fgets(command, 260, fp);
    command[strlen(command)-1]='\0';//去除换行符
    //读取工作文件夹
    fgets(workingdir, 260, fp);
    fclose(fp);
    ShellExecute(NULL,"open",command,NULL,workingdir,SW_SHOWNORMAL);
}