#include "Search.h"

//#define TEST

using namespace std;

#ifndef TEST
int main(int argc, char* argv[])
{
    char search_path[1000];
    char search_depth[50];
    char ignorePath[3000];
    char output[1000];
    char isIgnoreSearchDepth[2];

    if (argc == 6)
    {
        strcpy_s(search_path, argv[1]);
        strcpy_s(search_depth, argv[2]);
        setSearchDepth(atoi(search_depth));
        strcpy_s(ignorePath, argv[3]);
        strcpy_s(output, argv[4]);
        strcpy_s(isIgnoreSearchDepth, argv[5]);
     

        cout << "searchPath:" << search_path << endl;
        cout << "searchDepth:" << search_depth << endl;
        cout << "ignorePath:" << ignorePath << endl;
        cout << "output:" << output << endl;
        cout << "isIgnoreSearchDepth:" << isIgnoreSearchDepth << endl << endl;

        sqlite3_config(SQLITE_CONFIG_MULTITHREAD);
        sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 0);

        //打开数据库
        dbRes = sqlite3_open(output, &db);
        if (dbRes) {
            cout << "open database error" << endl << endl;
            exit(0);
        }
        cout << "open database successfully" << endl << endl;

        sqlite3_exec(db, "PRAGMA TEMP_STORE=MEMORY;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA journal_mode=OFF;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA cache_size=262144;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA page_size=65535;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA auto_vacuum=0;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA mmap_size=4096;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);

        char* p;
        char* _ignorepath = ignorePath;
        p = strtok(_ignorepath, ",");
        if (p != nullptr)
        {
            addIgnorePath(p);
        }
        
        while (p != nullptr)
        {
            p = strtok(nullptr, ",");
            if (p != nullptr)
            {
                addIgnorePath(p);
            }
        }

        cout << "ignorePathVector Size:" << ignorePathVector.size() << endl;

        initAllVector();
        if (atoi(isIgnoreSearchDepth) == 1)
        {
            cout << "ignore searchDepth!!!" << endl;
            searchFilesIgnoreSearchDepth(search_path, "*");
        }

        else
        {
            cout << "normal search" << endl;
            searchFiles(search_path, "*");
        }

        sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);

        executeAll(command0, "INSERT OR IGNORE INTO list0 VALUES(?, ?);");
        executeAll(command1, "INSERT OR IGNORE INTO list1 VALUES(?, ?);");
        executeAll(command2, "INSERT OR IGNORE INTO list2 VALUES(?, ?);");
        executeAll(command3, "INSERT OR IGNORE INTO list3 VALUES(?, ?);");
        executeAll(command4, "INSERT OR IGNORE INTO list4 VALUES(?, ?);");
        executeAll(command5, "INSERT OR IGNORE INTO list5 VALUES(?, ?);");
        executeAll(command6, "INSERT OR IGNORE INTO list6 VALUES(?, ?);");
        executeAll(command7, "INSERT OR IGNORE INTO list7 VALUES(?, ?);");
        executeAll(command8, "INSERT OR IGNORE INTO list8 VALUES(?, ?);");
        executeAll(command9, "INSERT OR IGNORE INTO list9 VALUES(?, ?);");
        executeAll(command10, "INSERT OR IGNORE INTO list10 VALUES(?, ?);");
        executeAll(command11, "INSERT OR IGNORE INTO list11 VALUES(?, ?);");
        executeAll(command12, "INSERT OR IGNORE INTO list12 VALUES(?, ?);");
        executeAll(command13, "INSERT OR IGNORE INTO list13 VALUES(?, ?);");
        executeAll(command14, "INSERT OR IGNORE INTO list14 VALUES(?, ?);");
        executeAll(command15, "INSERT OR IGNORE INTO list15 VALUES(?, ?);");
        executeAll(command16, "INSERT OR IGNORE INTO list16 VALUES(?, ?);");
        executeAll(command17, "INSERT OR IGNORE INTO list17 VALUES(?, ?);");
        executeAll(command18, "INSERT OR IGNORE INTO list18 VALUES(?, ?);");
        executeAll(command19, "INSERT OR IGNORE INTO list19 VALUES(?, ?);");
        executeAll(command20, "INSERT OR IGNORE INTO list20 VALUES(?, ?);");
        executeAll(command21, "INSERT OR IGNORE INTO list21 VALUES(?, ?);");
        executeAll(command22, "INSERT OR IGNORE INTO list22 VALUES(?, ?);");
        executeAll(command23, "INSERT OR IGNORE INTO list23 VALUES(?, ?);");
        executeAll(command24, "INSERT OR IGNORE INTO list24 VALUES(?, ?);");
        executeAll(command25, "INSERT OR IGNORE INTO list25 VALUES(?, ?);");
        executeAll(command26, "INSERT OR IGNORE INTO list26 VALUES(?, ?);");
        executeAll(command27, "INSERT OR IGNORE INTO list27 VALUES(?, ?);");
        executeAll(command28, "INSERT OR IGNORE INTO list28 VALUES(?, ?);");
        executeAll(command29, "INSERT OR IGNORE INTO list29 VALUES(?, ?);");
        executeAll(command30, "INSERT OR IGNORE INTO list30 VALUES(?, ?);");
        executeAll(command31, "INSERT OR IGNORE INTO list31 VALUES(?, ?);");
        executeAll(command32, "INSERT OR IGNORE INTO list32 VALUES(?, ?);");
        executeAll(command33, "INSERT OR IGNORE INTO list33 VALUES(?, ?);");
        executeAll(command34, "INSERT OR IGNORE INTO list34 VALUES(?, ?);");
        executeAll(command35, "INSERT OR IGNORE INTO list35 VALUES(?, ?);");
        executeAll(command36, "INSERT OR IGNORE INTO list36 VALUES(?, ?);");
        executeAll(command37, "INSERT OR IGNORE INTO list37 VALUES(?, ?);");
        executeAll(command38, "INSERT OR IGNORE INTO list38 VALUES(?, ?);");
        executeAll(command39, "INSERT OR IGNORE INTO list39 VALUES(?, ?);");
        executeAll(command40, "INSERT OR IGNORE INTO list40 VALUES(?, ?);");

        sqlite3_exec(db, "COMMIT;", 0, 0, 0);
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
    const char* output = "D:\\Code\\File-Engine\\File-Engine\\data.db";
    //打开数据库
    dbRes = sqlite3_open(output, &db);
    if (dbRes) {
        cout << "open database error" << endl << endl;
        exit(0);
    }
    cout << "open database successfully" << endl << endl;

    setSearchDepth(6);
    addIgnorePath("C:\\Windows");
    searchFiles("D:\\Code\\File-Engine\\File-Engine", "*");
    executeAll(command0, "INSERT OR IGNORE INTO list0 VALUES(?, ?);");
    executeAll(command1, "INSERT OR IGNORE INTO list1 VALUES(?, ?);");
    executeAll(command2, "INSERT OR IGNORE INTO list2 VALUES(?, ?);");
    executeAll(command3, "INSERT OR IGNORE INTO list3 VALUES(?, ?);");
    executeAll(command4, "INSERT OR IGNORE INTO list4 VALUES(?, ?);");
    executeAll(command5, "INSERT OR IGNORE INTO list5 VALUES(?, ?);");
    executeAll(command6, "INSERT OR IGNORE INTO list6 VALUES(?, ?);");
    executeAll(command7, "INSERT OR IGNORE INTO list7 VALUES(?, ?);");
    executeAll(command8, "INSERT OR IGNORE INTO list8 VALUES(?, ?);");
    executeAll(command9, "INSERT OR IGNORE INTO list9 VALUES(?, ?);");
    executeAll(command10, "INSERT OR IGNORE INTO list10 VALUES(?, ?);");
    executeAll(command11, "INSERT OR IGNORE INTO list11 VALUES(?, ?);");
    executeAll(command12, "INSERT OR IGNORE INTO list12 VALUES(?, ?);");
    executeAll(command13, "INSERT OR IGNORE INTO list13 VALUES(?, ?);");
    executeAll(command14, "INSERT OR IGNORE INTO list14 VALUES(?, ?);");
    executeAll(command15, "INSERT OR IGNORE INTO list15 VALUES(?, ?);");
    executeAll(command16, "INSERT OR IGNORE INTO list16 VALUES(?, ?);");
    executeAll(command17, "INSERT OR IGNORE INTO list17 VALUES(?, ?);");
    executeAll(command18, "INSERT OR IGNORE INTO list18 VALUES(?, ?);");
    executeAll(command19, "INSERT OR IGNORE INTO list19 VALUES(?, ?);");
    executeAll(command20, "INSERT OR IGNORE INTO list20 VALUES(?, ?);");
    executeAll(command21, "INSERT OR IGNORE INTO list21 VALUES(?, ?);");
    executeAll(command22, "INSERT OR IGNORE INTO list22 VALUES(?, ?);");
    executeAll(command23, "INSERT OR IGNORE INTO list23 VALUES(?, ?);");
    executeAll(command24, "INSERT OR IGNORE INTO list24 VALUES(?, ?);");
    executeAll(command25, "INSERT OR IGNORE INTO list25 VALUES(?, ?);");
    executeAll(command26, "INSERT OR IGNORE INTO list26 VALUES(?, ?);");
    executeAll(command27, "INSERT OR IGNORE INTO list27 VALUES(?, ?);");
    executeAll(command28, "INSERT OR IGNORE INTO list28 VALUES(?, ?);");
    executeAll(command29, "INSERT OR IGNORE INTO list29 VALUES(?, ?);");
    executeAll(command30, "INSERT OR IGNORE INTO list30 VALUES(?, ?);");
    executeAll(command31, "INSERT OR IGNORE INTO list31 VALUES(?, ?);");
    executeAll(command32, "INSERT OR IGNORE INTO list32 VALUES(?, ?);");
    executeAll(command33, "INSERT OR IGNORE INTO list33 VALUES(?, ?);");
    executeAll(command34, "INSERT OR IGNORE INTO list34 VALUES(?, ?);");
    executeAll(command35, "INSERT OR IGNORE INTO list35 VALUES(?, ?);");
    executeAll(command36, "INSERT OR IGNORE INTO list36 VALUES(?, ?);");
    executeAll(command37, "INSERT OR IGNORE INTO list37 VALUES(?, ?);");
    executeAll(command38, "INSERT OR IGNORE INTO list38 VALUES(?, ?);");
    executeAll(command39, "INSERT OR IGNORE INTO list39 VALUES(?, ?);");
    executeAll(command40, "INSERT OR IGNORE INTO list40 VALUES(?, ?);");
    sqlite3_close(db);
    getchar();
}

#endif
