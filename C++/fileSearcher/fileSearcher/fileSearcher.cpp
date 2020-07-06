#include "Search.h"

//#define TEST

using namespace std;

#ifndef TEST
int main(int argc, char* argv[])
{
    char searchPath[1000];
    char searchDepth[50];
    char ignorePath[3000];
    char output[1000];
    char isIgnoreSearchDepth[2];

    if (argc == 6)
    {
        strcpy_s(searchPath, argv[1]);
        strcpy_s(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy_s(ignorePath, argv[3]);
        strcpy_s(output, argv[4]);
        strcpy_s(isIgnoreSearchDepth, argv[5]);
     

        cout << "searchPath:" << searchPath << endl;
        cout << "searchDepth:" << searchDepth << endl;
        cout << "ignorePath:" << ignorePath << endl;
        cout << "output:" << output << endl;
        cout << "isIgnoreSearchDepth:" << isIgnoreSearchDepth << endl << endl;

        sqlite3_config(SQLITE_CONFIG_MULTITHREAD);

        //打开数据库
        dbRes = sqlite3_open(output, &db);
        if (dbRes) {
            cout << "open database error" << endl << endl;
            exit(0);
        }
        else {
            cout << "open database successfully" << endl << endl;
        }

        sqlite3_exec(db, "PRAGMA TEMP_STORE=MEMORY;", 0, 0, 0);
        sqlite3_exec(db, "PRAGMA journal_mode=OFF;", 0, 0, 0);
        sqlite3_exec(db, "PRAGMA cache_size=50000;", 0, 0, 0);
        sqlite3_exec(db, "PRAGMA page_size=8192;", 0, 0, 0);
        sqlite3_exec(db, "PRAGMA auto_vacuum=0;", 0, 0, 0);
        sqlite3_exec(db, "PRAGMA mmap_size=4096;", 0, 0, 0);
        sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);

        char* p = NULL;
        char* _ignorepath = ignorePath;
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

        executeAll(command0, "INSERT INTO list0(PATH) VALUES(?);");
        executeAll(command1, "INSERT INTO list1(PATH) VALUES(?);");
        executeAll(command2, "INSERT INTO list2(PATH) VALUES(?);");
        executeAll(command3, "INSERT INTO list3(PATH) VALUES(?);");
        executeAll(command4, "INSERT INTO list4(PATH) VALUES(?);");
        executeAll(command5, "INSERT INTO list5(PATH) VALUES(?);");
        executeAll(command6, "INSERT INTO list6(PATH) VALUES(?);");
        executeAll(command7, "INSERT INTO list7(PATH) VALUES(?);");
        executeAll(command8, "INSERT INTO list8(PATH) VALUES(?);");
        executeAll(command9, "INSERT INTO list9(PATH) VALUES(?);");
        executeAll(command10, "INSERT INTO list10(PATH) VALUES(?);");
        executeAll(command11, "INSERT INTO list11(PATH) VALUES(?);");
        executeAll(command12, "INSERT INTO list12(PATH) VALUES(?);");
        executeAll(command13, "INSERT INTO list13(PATH) VALUES(?);");
        executeAll(command14, "INSERT INTO list14(PATH) VALUES(?);");
        executeAll(command15, "INSERT INTO list15(PATH) VALUES(?);");
        executeAll(command16, "INSERT INTO list16(PATH) VALUES(?);");
        executeAll(command17, "INSERT INTO list17(PATH) VALUES(?);");
        executeAll(command18, "INSERT INTO list18(PATH) VALUES(?);");
        executeAll(command19, "INSERT INTO list19(PATH) VALUES(?);");
        executeAll(command20, "INSERT INTO list20(PATH) VALUES(?);");
        executeAll(command21, "INSERT INTO list21(PATH) VALUES(?);");
        executeAll(command22, "INSERT INTO list22(PATH) VALUES(?);");
        executeAll(command23, "INSERT INTO list23(PATH) VALUES(?);");
        executeAll(command24, "INSERT INTO list24(PATH) VALUES(?);");
        executeAll(command25, "INSERT INTO list25(PATH) VALUES(?);");
        executeAll(command26, "INSERT INTO list26(PATH) VALUES(?);");
        executeAll(command27, "INSERT INTO list27(PATH) VALUES(?);");
        executeAll(command28, "INSERT INTO list28(PATH) VALUES(?);");
        executeAll(command29, "INSERT INTO list29(PATH) VALUES(?);");
        executeAll(command30, "INSERT INTO list30(PATH) VALUES(?);");
        executeAll(command31, "INSERT INTO list31(PATH) VALUES(?);");
        executeAll(command32, "INSERT INTO list32(PATH) VALUES(?);");
        executeAll(command33, "INSERT INTO list33(PATH) VALUES(?);");
        executeAll(command34, "INSERT INTO list34(PATH) VALUES(?);");
        executeAll(command35, "INSERT INTO list35(PATH) VALUES(?);");
        executeAll(command36, "INSERT INTO list36(PATH) VALUES(?);");
        executeAll(command37, "INSERT INTO list37(PATH) VALUES(?);");
        executeAll(command38, "INSERT INTO list38(PATH) VALUES(?);");
        executeAll(command39, "INSERT INTO list39(PATH) VALUES(?);");
        executeAll(command40, "INSERT INTO list40(PATH) VALUES(?);");
        //创建索引
        sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
        char num[5];
        string str;
        for (int i = 0; i < 40; i++) {
            _itoa_s(i, num, 10);
            str.append("CREATE UNIQUE INDEX list").append(num).append("_index ON list").append(num).append("(PATH)").append(";");
            sqlite3_exec(db, str.c_str(), NULL, NULL, NULL);
            str.clear();
        }
        sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
        sqlite3_close(db);
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
    const char* output = "C:\\Users\\13927\\Desktop\\test.db";
    //打开数据库
    dbRes = sqlite3_open(output, &db);
    if (dbRes) {
        cout << "open database error" << endl << endl;
        exit(0);
    }
    else {
        cout << "open database successfully" << endl << endl;
    }

    setSearchDepth(6);
    addIgnorePath("C:\\Windows");
    searchFiles("C:", "*");
    executeAll(command0, "INSERT INTO list0(PATH) VALUES(?);");
    executeAll(command1, "INSERT INTO list1(PATH) VALUES(?);");
    executeAll(command2, "INSERT INTO list2(PATH) VALUES(?);");
    executeAll(command3, "INSERT INTO list3(PATH) VALUES(?);");
    executeAll(command4, "INSERT INTO list4(PATH) VALUES(?);");
    executeAll(command5, "INSERT INTO list5(PATH) VALUES(?);");
    executeAll(command6, "INSERT INTO list6(PATH) VALUES(?);");
    executeAll(command7, "INSERT INTO list7(PATH) VALUES(?);");
    executeAll(command8, "INSERT INTO list8(PATH) VALUES(?);");
    executeAll(command9, "INSERT INTO list9(PATH) VALUES(?);");
    executeAll(command10, "INSERT INTO list10(PATH) VALUES(?);");
    executeAll(command11, "INSERT INTO list11(PATH) VALUES(?);");
    executeAll(command12, "INSERT INTO list12(PATH) VALUES(?);");
    executeAll(command13, "INSERT INTO list13(PATH) VALUES(?);");
    executeAll(command14, "INSERT INTO list14(PATH) VALUES(?);");
    executeAll(command15, "INSERT INTO list15(PATH) VALUES(?);");
    executeAll(command16, "INSERT INTO list16(PATH) VALUES(?);");
    executeAll(command17, "INSERT INTO list17(PATH) VALUES(?);");
    executeAll(command18, "INSERT INTO list18(PATH) VALUES(?);");
    executeAll(command19, "INSERT INTO list19(PATH) VALUES(?);");
    executeAll(command20, "INSERT INTO list20(PATH) VALUES(?);");
    executeAll(command21, "INSERT INTO list21(PATH) VALUES(?);");
    executeAll(command22, "INSERT INTO list22(PATH) VALUES(?);");
    executeAll(command23, "INSERT INTO list23(PATH) VALUES(?);");
    executeAll(command24, "INSERT INTO list24(PATH) VALUES(?);");
    executeAll(command25, "INSERT INTO list25(PATH) VALUES(?);");
    executeAll(command26, "INSERT INTO list26(PATH) VALUES(?);");
    executeAll(command27, "INSERT INTO list27(PATH) VALUES(?);");
    executeAll(command28, "INSERT INTO list28(PATH) VALUES(?);");
    executeAll(command29, "INSERT INTO list29(PATH) VALUES(?);");
    executeAll(command30, "INSERT INTO list30(PATH) VALUES(?);");
    executeAll(command31, "INSERT INTO list31(PATH) VALUES(?);");
    executeAll(command32, "INSERT INTO list32(PATH) VALUES(?);");
    executeAll(command33, "INSERT INTO list33(PATH) VALUES(?);");
    executeAll(command34, "INSERT INTO list34(PATH) VALUES(?);");
    executeAll(command35, "INSERT INTO list35(PATH) VALUES(?);");
    executeAll(command36, "INSERT INTO list36(PATH) VALUES(?);");
    executeAll(command37, "INSERT INTO list37(PATH) VALUES(?);");
    executeAll(command38, "INSERT INTO list38(PATH) VALUES(?);");
    executeAll(command39, "INSERT INTO list39(PATH) VALUES(?);");
    executeAll(command40, "INSERT INTO list40(PATH) VALUES(?);");
    sqlite3_close(db);
    getchar();
}

#endif
