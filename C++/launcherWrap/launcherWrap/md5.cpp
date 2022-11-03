#include "md5.h"
#include <iomanip>
#include <sstream>
#pragma comment(lib, "Advapi32.lib")

void ShowError(const TCHAR* pszText)
{
	wchar_t szErr[MAX_PATH] = { 0 };
	::wsprintf(szErr, L"%s Error[%d]\n", pszText, GetLastError());
}

BOOL GetFileData(const TCHAR* pszFilePath, BYTE** ppFileData, DWORD* pdwFileDataLength)
{
	BOOL bRet = TRUE;
	HANDLE hFile;
	DWORD dwTemp = 0;

	do
	{
		hFile = ::CreateFile(pszFilePath, GENERIC_READ | GENERIC_WRITE,
			FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
			FILE_ATTRIBUTE_ARCHIVE, nullptr);
		if (INVALID_HANDLE_VALUE == hFile)
		{
			bRet = FALSE;
			ShowError(L"CreateFile");
			break;
		}

		const DWORD dwFileDataLength = GetFileSize(hFile, nullptr);

		BYTE* pFileData = new BYTE[dwFileDataLength];
		if (nullptr == pFileData)
		{
			bRet = FALSE;
			ShowError(L"new");
			break;
		}
		::RtlZeroMemory(pFileData, dwFileDataLength);

		ReadFile(hFile, pFileData, dwFileDataLength, &dwTemp, nullptr);

		// 返回
		*ppFileData = pFileData;
		*pdwFileDataLength = dwFileDataLength;

	} while (FALSE);

	if (hFile)
	{
		CloseHandle(hFile);
	}

	return bRet;
}

BOOL CalculateHash(BYTE* pData, DWORD dwDataLength, ALG_ID algHashType, BYTE** ppHashData, DWORD* pdwHashDataLength)
{
	HCRYPTPROV hCryptProv = NULL;
	HCRYPTHASH hCryptHash = NULL;
	BYTE* pHashData = nullptr;
	DWORD dwHashDataLength = 0;
	DWORD dwTemp;
	BOOL bRet;

	do
	{
		// 获得指定CSP的密钥容器的句柄
		bRet = ::CryptAcquireContext(&hCryptProv, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT);
		if (FALSE == bRet)
		{
			ShowError(L"CryptAcquireContext");
			break;
		}

		// 创建一个HASH对象, 指定HASH算法
		bRet = CryptCreateHash(hCryptProv, algHashType, NULL, NULL, &hCryptHash);
		if (FALSE == bRet)
		{
			ShowError(L"CryptCreateHash");
			break;
		}

		// 计算HASH数据
		bRet = CryptHashData(hCryptHash, pData, dwDataLength, 0);
		if (FALSE == bRet)
		{
			ShowError(L"CryptHashData");
			break;
		}

		// 获取HASH结果的大小
		dwTemp = sizeof(dwHashDataLength);
		bRet = CryptGetHashParam(hCryptHash, HP_HASHSIZE, (BYTE*)(&dwHashDataLength), &dwTemp, 0);
		if (FALSE == bRet)
		{
			ShowError(L"CryptGetHashParam");
			break;
		}

		// 申请内存
		pHashData = new BYTE[dwHashDataLength];
		if (nullptr == pHashData)
		{
			bRet = FALSE;
			ShowError(L"new");
			break;
		}
		::RtlZeroMemory(pHashData, dwHashDataLength);

		// 获取HASH结果数据
		bRet = CryptGetHashParam(hCryptHash, HP_HASHVAL, pHashData, &dwHashDataLength, 0);
		if (FALSE == bRet)
		{
			ShowError(L"CryptGetHashParam");
			break;
		}

		// 返回数据
		*ppHashData = pHashData;
		*pdwHashDataLength = dwHashDataLength;

	} while (FALSE);

	// 释放关闭
	if (FALSE == bRet)
	{
		if (pHashData)
		{
			delete[]pHashData;
			pHashData = nullptr;
		}
	}
	if (hCryptHash)
	{
		CryptDestroyHash(hCryptHash);
	}
	if (hCryptProv)
	{
		CryptReleaseContext(hCryptProv, 0);
	}

	return bRet;
}

std::string hexStr(const uint8_t* data, const unsigned len)
{
	std::stringstream ss;
	ss << std::hex;

	for (unsigned i(0); i < len; ++i)
		ss << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);

	return ss.str();
}

std::string GetFileHash(const TCHAR* filePath)
{
	BYTE* pData = nullptr;
	DWORD dwDataLength = 0;
	BYTE* pHashData = nullptr;
	DWORD dwHashDataLength = 0;

	// 读取文件数据
	GetFileData(filePath, &pData, &dwDataLength);

	// MD5
	CalculateHash(pData, dwDataLength, CALG_MD5, &pHashData, &dwHashDataLength);

	auto&& ret = hexStr(pHashData, dwHashDataLength);

	delete[] pData;
	delete[] pHashData;
	return ret;
}