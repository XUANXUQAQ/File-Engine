#include <stdio.h>
#include <windows.h>


int main(){
    HWND hwnd;  
    hwnd=FindWindow("ConsoleWindowClass",NULL); //处理顶级窗口的类名和窗口名称匹配指定的字符串,不搜索子窗口。  
    if(hwnd)  
    {  
        ShowWindow(hwnd,SW_HIDE);               //设置指定窗口的显示状态  
    }  
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
