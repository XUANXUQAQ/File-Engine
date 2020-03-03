package everythingSDK;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.WString;

import java.nio.Buffer;

public interface Everything extends Library {
    Everything INSTANCE = Native.load("Everything", Everything.class);


    int EVERYTHING_OK = 0;
    int EVERYTHING_ERROR_MEMORY = 1;
    int EVERYTHING_ERROR_IPC = 2;
    int EVERYTHING_ERROR_REGISTERCLASSEX = 3;
    int EVERYTHING_ERROR_CREATEWINDOW = 4;
    int EVERYTHING_ERROR_CREATETHREAD = 5;
    int EVERYTHING_ERROR_INVALIDINDEX = 6;
    int EVERYTHING_ERROR_INVALIDCALL = 7;

    int Everything_SetSearchW(WString lpSearchString);


    void Everything_SetMatchPath(boolean bEnable);


    void Everything_SetMatchCase(boolean bEnable);


    void Everything_SetMatchWholeWord(boolean bEnable);


    void Everything_SetRegex(boolean bEnable);


    void Everything_SetMax(int dwMax);


    void Everything_SetOffset(int dwOffset);


    boolean Everything_GetMatchPath();


    boolean Everything_GetMatchCase();


    boolean Everything_GetMatchWholeWord();


    boolean Everything_GetRegex();


    int Everything_GetMax();


    int Everything_GetOffset();


    WString Everything_GetSearchW();


    int Everything_GetLastError();


    boolean Everything_QueryW(boolean bWait);


    void Everything_SortResultsByPath();


    int Everything_GetNumFileResults();


    int Everything_GetNumFolderResults();


    int Everything_GetNumResults();


    int Everything_GetTotFileResults();


    int Everything_GetTotFolderResults();


    int Everything_GetTotResults();


    boolean Everything_IsVolumeResult(int nIndex);


    boolean Everything_IsFolderResult(int nIndex);


    boolean Everything_IsFileResult(int nIndex);


    void Everything_GetResultFullPathNameW(int nIndex, Buffer lpString, int nMaxCount);


    void Everything_Reset();

}
