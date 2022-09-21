#pragma once

#define UTILITIES_REGEX
#define UTILITIES_FILE
#define CONSOLE_WIDTH 79
#define UTILITIES_NO_CPP17

#pragma warning(disable:26451)
#pragma warning(disable:6386)
#include "constans.h"
#include <cmath>
#include <vector>
#ifdef UTILITIES_REGEX
#include <regex> // contains <string>, <vector>, <algorithm> and others
#else // UTILITIES_REGEX
#include <string>
#endif // UTILITIES_REGEX
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif
#include <thread> // contains <chrono>
#undef min
#undef max
using std::string;
using std::vector;
using std::thread;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef int64_t slong;
typedef uint64_t ulong;
#define pif 3.1415927f
#define min_char ((char)-128)
#define max_char ((char)127)
#define max_uchar ((uchar)255)
#define min_short ((short)-32768)
#define max_short ((short)32767)
#define max_ushort ((ushort)65535)
#define min_int -2147483648
#define max_int 2147483647
#define max_uint 4294967295u
#define min_slong -9223372036854775808ll
#define max_slong 9223372036854775807ll
#define max_ulong 18446744073709551615ull
#define min_float 1.401298464E-45f
#define max_float 3.402823466E38f
#define epsilon_float 1.192092896E-7f
#define min_double 4.9406564584124654E-324
#define max_double 1.7976931348623158E308
#define epsilon_double 2.2204460492503131E-16

class Clock {
private:
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> t;
public:
	Clock() { start(); }
	void start() { t = clock::now(); }
	double stop() const { return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now()-t).count(); }
};
inline void sleep(const double t) {
	if(t>0.0) std::this_thread::sleep_for(std::chrono::milliseconds((int)(1E3*t+0.5)));
}

inline float as_float(const uint x) {
	return *(float*)&x;
}
inline uint as_uint(const float x) {
	return *(uint*)&x;
}
inline double as_double(const ulong x) {
	return *(double*)&x;
}
inline ulong as_ulong(const double x) {
	return *(ulong*)&x;
}

inline float half_to_float(const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	const uint e = (x&0x7C00)>>10; // exponent
	const uint m = (x&0x03FF)<<13; // mantissa
	const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
	return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}
inline ushort float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	const uint b = as_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
	const uint e = (b&0x7F800000)>>23; // exponent
	const uint m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
	return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
}

inline float sq(const float x) {
	return x*x;
}
inline float cb(const float x) {
	return x*x*x;
}
inline float pow(const float x, const uint n) {
	float r = 1.0f;
	for(uint i=0u; i<n; i++) {
		r *= x;
	}
	return r;
}
inline float sign(const float x) {
	return x>=0.0f ? 1.0f : -1.0f;
}
inline float clamp(const float x, const float a, const float b) {
	return fmin(fmax(x, a), b);
}
inline float rsqrt(const float x) {
	return 1.0f/sqrt(x);
}
inline float ln(const float x) {
	return log(x); // natural logarithm
}
inline float random(const float x=1.0f) {
	return x*((float)rand()/(float)RAND_MAX);
}
inline float random_symmetric(const float x=1.0f) {
	return 2.0f*x*((float)rand()/(float)RAND_MAX-0.5f);
}

inline double sq(const double x) {
	return x*x;
}
inline double cb(const double x) {
	return x*x*x;
}
inline double pow(const double x, const uint n) {
	double r = 1.0;
	for(uint i=0u; i<n; i++) {
		r *= x;
	}
	return r;
}
inline double sign(const double x) {
	return x>=0.0 ? 1.0 : -1.0;
}
inline double clamp(const double x, const double a, const double b) {
	return fmin(fmax(x, a), b);
}
inline double rsqrt(const double x) {
	return 1.0/sqrt(x);
}
inline double ln(const double x) {
	return log(x); // natural logarithm
}

inline int sq(const int x) {
	return x*x;
}
inline int cb(const int x) {
	return x*x*x;
}
inline int pow(const int x, const uint n) {
	int r = 1;
	for(uint i=0u; i<n; i++) {
		r *= x;
	}
	return r;
}
inline int sign(const int x) {
	return 1-2*(x>>31&1);
}
inline int min(const int x, const int y) {
	return x<y?x:y;
}
inline int max(const int x, const int y) {
	return x>y?x:y;
}
inline int clamp(const int x, const int a, const int b) {
	return min(max(x, a), b);
}

inline uint sq(const uint x) {
	return x*x;
}
inline uint cb(const uint x) {
	return x*x*x;
}
inline uint pow(const uint x, const uint n) {
	uint r = 1u;
	for(uint i=0u; i<n; i++) {
		r *= x;
	}
	return r;
}
inline uint min(const uint x, const uint y) {
	return x<y?x:y;
}
inline uint max(const uint x, const uint y) {
	return x>y?x:y;
}
inline uint clamp(const uint x, const uint a, const uint b) {
	return min(max(x, a), b);
}
inline uint gcd(uint x, uint y) { // greatest common divisor
	if(x*y==0u) return 0u;
	uint t;
	while(y!=0u) {
		t = y;
		y = x%y;
		x = t;
	}
	return x;
}
inline uint lcm(const uint x, const uint y) { // least common multiple
	return x*y==0u ? 0u : x*y/gcd(x, y);
}

inline slong sq(const slong x) {
	return x*x;
}
inline slong cb(const slong x) {
	return x*x*x;
}
inline slong pow(const slong x, const uint n) {
	slong r = 1ll;
	for(uint i=0u; i<n; i++) {
		r *= x;
	}
	return r;
}
inline slong sign(const slong x) {
	return 1ll-2ll*(x>>63&1ll);
}
inline slong min(const slong x, const slong y) {
	return x<y?x:y;
}
inline slong max(const slong x, const slong y) {
	return x>y?x:y;
}
inline slong clamp(const slong x, const slong a, const slong b) {
	return min(max(x, a), b);
}

inline ulong sq(const ulong x) {
	return x*x;
}
inline ulong cb(const ulong x) {
	return x*x*x;
}
inline ulong pow(const ulong x, const uint n) {
	ulong r = 1ull;
	for(uint i=0u; i<n; i++) {
		r *= x;
	}
	return r;
}
inline ulong min(const ulong x, const ulong y) {
	return x<y?x:y;
}
inline ulong max(const ulong x, const ulong y) {
	return x>y?x:y;
}
inline ulong clamp(const ulong x, const ulong a, const ulong b) {
	return min(max(x, a), b);
}
inline ulong gcd(ulong x, ulong y) { // greatest common divisor
	if(x*y==0ull) return 0ull;
	ulong t;
	while(y!=0ull) {
		t = y;
		y = x%y;
		x = t;
	}
	return x;
}
inline ulong lcm(const ulong x, const ulong y) { // least common multiple
	return x*y==0ull ? 0ull : x*y/gcd(x, y);
}

inline int to_int(const float x) {
	return (int)(x+0.5f-(float)(x<0.0f));
}
inline int to_int(const double x) {
	return (int)(x+0.5-(double)(x<0.0));
}
inline uint to_uint(const float x) {
	return (uint)fmax(x+0.5f, 0.5f);
}
inline uint to_uint(const double x) {
	return (uint)fmax(x+0.5, 0.5);
}
inline slong to_slong(const float x) {
	return (slong)(x+0.5f);
}
inline slong to_slong(const double x) {
	return (slong)(x+0.5);
}
inline ulong to_ulong(const float x) {
	return (ulong)fmax(x+0.5f, 0.5f);
}
inline ulong to_ulong(const double x) {
	return (ulong)fmax(x+0.5, 0.5);
}

inline void split_float(float x, uint& integral, uint& decimal, int& exponent) {
	if(x>=10.0f) { // convert to base 10
		if(x>=1E32f) { x *= 1E-32f; exponent += 32; }
		if(x>=1E16f) { x *= 1E-16f; exponent += 16; }
		if(x>= 1E8f) { x *=  1E-8f; exponent +=  8; }
		if(x>= 1E4f) { x *=  1E-4f; exponent +=  4; }
		if(x>= 1E2f) { x *=  1E-2f; exponent +=  2; }
		if(x>= 1E1f) { x *=  1E-1f; exponent +=  1; }
	}
	if(x>0.0f && x<=1.0f) {
		if(x<1E-31f) { x *=  1E32f; exponent -= 32; }
		if(x<1E-15f) { x *=  1E16f; exponent -= 16; }
		if(x< 1E-7f) { x *=   1E8f; exponent -=  8; }
		if(x< 1E-3f) { x *=   1E4f; exponent -=  4; }
		if(x< 1E-1f) { x *=   1E2f; exponent -=  2; }
		if(x<  1E0f) { x *=   1E1f; exponent -=  1; }
	}
	integral = (uint)x;
	float remainder = (x-integral)*1E8f; // 8 decimal digits
	decimal = (uint)remainder;
	if(remainder-(float)decimal>=0.5f) { // correct rounding of last decimal digit
		decimal++;
		if(decimal>=100000000u) { // decimal overflow
			decimal = 0u;
			integral++;
			if(integral>=10u) { // decimal overflow causes integral overflow
				integral = 1u;
				exponent++;
			}
		}
	}
}
inline void split_double(double x, uint& integral, ulong& decimal, int& exponent) {
	if(x>=10.0) { // convert to base 10
		if(x>=1E256) { x *= 1E-256; exponent += 256; }
		if(x>=1E128) { x *= 1E-128; exponent += 128; }
		if(x>= 1E64) { x *=  1E-64; exponent +=  64; }
		if(x>= 1E32) { x *=  1E-32; exponent +=  32; }
		if(x>= 1E16) { x *=  1E-16; exponent +=  16; }
		if(x>=  1E8) { x *=   1E-8; exponent +=   8; }
		if(x>=  1E4) { x *=   1E-4; exponent +=   4; }
		if(x>=  1E2) { x *=   1E-2; exponent +=   2; }
		if(x>=  1E1) { x *=   1E-1; exponent +=   1; }
	}
	if(x>0.0 && x<=1.0) {
		if(x<1E-255) { x *=  1E256; exponent -= 256; }
		if(x<1E-127) { x *=  1E128; exponent -= 128; }
		if(x< 1E-63) { x *=   1E64; exponent -=  64; }
		if(x< 1E-31) { x *=   1E32; exponent -=  32; }
		if(x< 1E-15) { x *=   1E16; exponent -=  16; }
		if(x<  1E-7) { x *=    1E8; exponent -=   8; }
		if(x<  1E-3) { x *=    1E4; exponent -=   4; }
		if(x<  1E-1) { x *=    1E2; exponent -=   2; }
		if(x<   1E0) { x *=    1E1; exponent -=   1; }
	}
	integral = (uint)x;
	double remainder = (x-integral)*1E16; // 16 decimal digits
	decimal = (ulong)remainder;
	if(remainder-(double)decimal>=0.5) { // correct rounding of last decimal digit
		decimal++;
		if(decimal>=10000000000000000ull) { // decimal overflow
			decimal = 0ull;
			integral++;
			if(integral>=10u) { // decimal overflow causes integral overflow
				integral = 1u;
				exponent++;
			}
		}
	}
}
inline string decimal_to_string_float(uint x, int digits) {
	string r = "";
	while((digits--)>0) {
		r = (char)(x%10u+48u)+r;
		x /= 10u;
	}
	return r;
}
inline string decimal_to_string_double(ulong x, int digits) {
	string r = "";
	while((digits--)>0) {
		r = (char)(x%10ull+48ull)+r;
		x /= 10ull;
	}
	return r;
}

inline string to_string(const string& s){
	return s;
}
inline string to_string(const char& c) {
	return string(1, c);
}
inline string to_string(ulong x) {
	string r = "";
	do {
		r = (char)(x%10ull+48ull)+r;
		x /= 10ull;
	} while(x);
	return r;
}
inline string to_string(slong x) {
	return x>=0ll ? to_string((ulong)x) : "-"+to_string((ulong)(-x));
}
inline string to_string(uint x) {
	string r = "";
	do {
		r = (char)(x%10u+48u)+r;
		x /= 10u;
	} while(x);
	return r;
}
inline string to_string(int x) {
	return x>=0 ? to_string((uint)x) : "-"+to_string((uint)(-x));
}
inline string to_string(float x) { // convert float to string with full precision (<string> to_string() prints only 6 decimals)
	string s = "";
	if(x<0.0f) { s += "-"; x = -x; }
	if(std::isnan(x)) return s+"NaN";
	if(std::isinf(x)) return s+"Inf";
	uint integral, decimal;
	int exponent = 0;
	split_float(x, integral, decimal, exponent);
	return s+to_string(integral)+"."+decimal_to_string_float(decimal, 8)+(exponent!=0?"E"+to_string(exponent):"");
}
inline string to_string(double x) { // convert double to string with full precision (<string> to_string() prints only 6 decimals)
	string s = "";
	if(x<0.0) { s += "-"; x = -x; }
	if(std::isnan(x)) return s+"NaN";
	if(std::isinf(x)) return s+"Inf";
	uint integral;
	ulong decimal;
	int exponent = 0;
	split_double(x, integral, decimal, exponent);
	return s+to_string(integral)+"."+decimal_to_string_double(decimal, 16)+(exponent!=0?"E"+to_string(exponent):"");
}
inline string to_string(float x, const uint decimals) { // convert float to string with specified number of decimals
	string s = "";
	if(x<0.0f) { s += "-"; x = -x; }
	if(std::isnan(x)) return s+"NaN";
	if(std::isinf(x)||x>(float)max_ulong) return s+"Inf";
	const float power = pow(10.0f, min(decimals, 8u));
	x += 0.5f/power; // rounding
	const ulong integral = (ulong)x;
	const uint decimal = (uint)((x-(float)integral)*power);
	return s+to_string(integral)+(decimals==0u ? "" : "."+decimal_to_string_float(decimal, min((int)decimals, 8)));
}
inline string to_string(double x, const uint decimals) { // convert float to string with specified number of decimals
	string s = "";
	if(x<0.0) { s += "-"; x = -x; }
	if(std::isnan(x)) return s+"NaN";
	if(std::isinf(x)||x>(double)max_ulong) return s+"Inf";
	const double power = pow(10.0, min(decimals, 16u));
	x += 0.5/power; // rounding
	const ulong integral = (ulong)x;
	const ulong decimal = (ulong)((x-(double)integral)*power);
	return s+to_string(integral)+(decimals==0u ? "" : "."+decimal_to_string_double(decimal, min((int)decimals, 16)));
}

inline uint length(const string& s) {
	return (uint)s.length();
}
inline bool contains(const string& s, const string& match) {
	return s.find(match)!=string::npos;
}
inline bool contains_any(const string& s, const vector<string>& matches) {
	for(uint i=0u; i<(uint)matches.size(); i++) if(contains(s, matches[i])) return true;
	return false;
}
inline string to_lower(const string& s) {
	string r = "";
	for(uint i=0u; i<(uint)s.length(); i++) {
		const uchar c = s.at(i);
		r += c>64u&&c<91u ? c+32u : c;
	}
	return r;
}
inline string to_upper(const string& s) {
	string r = "";
	for(uint i=0u; i<(uint)s.length(); i++) {
		const uchar c = s.at(i);
		r += c>96u&&c<123u ? c-32u : c;
	}
	return r;
}
inline bool equals(const string& a, const string& b) {
	return to_lower(a)==to_lower(b);
}
inline string replace(const string& s, const string& from, const string& to) {
	string r = s;
	int p = 0;
	while((p=(int)r.find(from, p))!=string::npos) {
		r.replace(p, from.length(), to);
		p += (int)to.length();
	}
	return r;
}
inline string substring(const string& s, const uint start, uint length=max_uint) {
	return s.substr(start, min(length, (uint)s.length()-start));
}
inline string trim(const string& s) { // removes whitespace characters from beginnig and end of string s
	const int l = (int)s.length();
	int a=0, b=l-1;
	char c;
	while(a<l && ((c=s[a])==' '||c=='\t'||c=='\n'||c=='\v'||c=='\f'||c=='\r'||c=='\0')) a++;
	while(b>a && ((c=s[b])==' '||c=='\t'||c=='\n'||c=='\v'||c=='\f'||c=='\r'||c=='\0')) b--;
	return s.substr(a, 1+b-a);
}
inline bool begins_with(const string& s, const string& match) {
	if(match.size()>s.size()) return false;
	else return equal(match.begin(), match.end(), s.begin());
}
inline bool ends_with(const string& s, const string& match) {
	if(match.size()>s.size()) return false;
	else return equal(match.rbegin(), match.rend(), s.rbegin());
}
template<class T> inline bool contains(const vector<T>& v, const T& match) {
	return find(v.begin(), v.end(), match)!=v.end();
}

inline string alignl(const uint n, const string& x="") { // converts x to string with spaces behind such that length is n if x is not longer than n
	string s = x;
	for(uint i=0u; i<n; i++) s += " ";
	return s.substr(0, max(n, (uint)x.length()));
}
inline string alignr(const uint n, const string& x="") { // converts x to string with spaces in front such that length is n if x is not longer than n
	string s = "";
	for(uint i=0u; i<n; i++) s += " ";
	s += x;
	return s.substr((uint)min((int)s.length()-(int)n, (int)n), s.length());
}
template<typename T> inline string alignl(const uint n, const T x) { // converts x to string with spaces behind such that length is n if x does not have more digits than n
	return alignl(n, to_string(x));
}
template<typename T> inline string alignr(const uint n, const T x) { // converts x to string with spaces in front such that length is n if x does not have more digits than n
	return alignr(n, to_string(x));
}

inline void print(const string& s="") {
#ifdef DEBUG_OUTPUT
	std::cout << s;
#endif
}
inline void println(const string& s="") {
#ifdef DEBUG_OUTPUT
	std::cout << s+'\n';
#endif
}
inline void reprint(const string& s="") {
#ifdef DEBUG_OUTPUT
	std::cout << "\r"+s;
#endif
}
inline void wait() {
#ifdef DEBUG_OUTPUT
	std::cin.get();
#endif
}
template<typename T> inline void println(const T x) {
	println(to_string(x));
}

#ifdef UTILITIES_REGEX
inline vector<string> split_regex(const string& s, const string& separator="\\s+") {
	vector<string> r;
	const std::regex rgx(separator);
	std::sregex_token_iterator token(s.begin(), s.end()+1, rgx, -1), end;
	while(token!=end) {
		r.push_back(*token);
		token++;
	}
	return r;
}
inline bool equals_regex(const string& s, const string& match) { // returns true if string exactly matches regex
	return regex_match(s.begin(), s.end(), std::regex(match));
}
inline uint matches_regex(const string& s, const string& match) { // counts number of matches
	std::regex words_regex(match);
	auto words_begin = std::sregex_iterator(s.begin(), s.end(), words_regex);
	auto words_end = std::sregex_iterator();
	return (uint)std::distance(words_begin, words_end);
}
inline bool contains_regex(const string& s, const string& match) {
	return matches_regex(s, match)>=1;
}
inline string replace_regex(const string& s, const string& from, const string& to) {
	return regex_replace(s, std::regex(from), to);
}
inline bool is_number(const string& s) {
	return equals_regex(s, "\\d+(u|l|ul|ll|ull)?")||equals_regex(s, "0x(\\d|[a-fA-F])+(u|l|ul|ll|ull)?")||equals_regex(s, "0b[01]+(u|l|ul|ll|ull)?")||equals_regex(s, "(((\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+[fF]?)?)|(\\d+\\.\\d*|\\.\\d+)[fF]?)");
}
inline void print_message(const string& message, const string& keyword="") { // print formatted message
	const uint k=length(keyword), w=CONSOLE_WIDTH-4u-k;
	uint l = 0u;
	string p="\r| "+keyword, f=" ";
	for(uint j=0u; j<k; j++) f += " ";
	vector<string> v = split_regex(message, "[\\s\\0]+");
	for(uint i=0u; i<(uint)v.size(); i++) {
		const string word = v.at(i);
		const uint wordlength = length(word);
		l += wordlength+1u;
		if(l<=w+1u||wordlength>w) {
			p += word+" ";
		} else {
			l = l-length(v.at(i--))-1u;
			for(uint j=l; j<=w; j++) p += " ";
			p += "|\n|"+f;
			l = 0u;
		}
	}
	for(uint j=l; j<=w; j++) p += " ";
	println(p+"|");
}
inline void print_error(const string& s) { // print formatted error message
	print_message(s, "Error: ");
#ifdef _WIN32
	print_message("Press Enter to exit.", "       ");
#endif // _WIN32
	string b = "";
	for(int i=0; i<CONSOLE_WIDTH-2; i++) b += "-";
	println("'"+b+"'");
#ifdef _WIN32
	wait();
#endif //_WIN32
	std::quick_exit(1);
}
inline void print_warning(const string& s) { // print formatted warning message
	print_message(s, "Warning: ");
}
inline void print_info(const string& s) { // print formatted info message
	print_message(s, "Info: ");
}

inline void parse_sanity_check_error(const string& s, const string& regex, const string& type) {
	if(!equals_regex(s, regex)) print_error("\""+s+"\" cannot be parsed to "+type+".");
}
inline int to_int(const string& s) {
	const string t = trim(s);
	parse_sanity_check_error(t, "[+-]?\\d+", "int");
	return atoi(t.c_str());
}
inline uint to_uint(const string& s) {
	const string t = trim(s);
	parse_sanity_check_error(t, "\\+?\\d+", "uint");
	return (uint)atoi(t.c_str());
}
inline slong to_slong(const string& s) {
	const string t = trim(s);
	parse_sanity_check_error(t, "[+-]?\\d+", "slong");
	return (slong)atoll(t.c_str());
}
inline ulong to_ulong(const string& s) {
	const string t = trim(s);
	parse_sanity_check_error(t, "\\+?\\d+", "ulong");
	return (ulong)atoll(t.c_str());
}
inline float to_float(const string& s) {
	const string t = trim(s);
	parse_sanity_check_error(t, "[+-]?(((\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+[fF]?)?)|(\\d+\\.\\d*|\\.\\d+)[fF]?)", "float");
	return (float)atof(t.c_str());
}
inline double to_double(const string& s) {
	const string t = trim(s);
	parse_sanity_check_error(t, "[+-]?(((\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+[fF]?)?)|(\\d+\\.\\d*|\\.\\d+)[fF]?)", "double");
	return atof(t.c_str());
}

inline bool parse_sanity_check(const string& s, const string& regex) {
	return equals_regex(s, regex);
}
inline int to_int(const string& s, const int default_value) {
	const string t = trim(s);
	return parse_sanity_check(t, "[+-]?\\d+") ? atoi(t.c_str()) : default_value;
}
inline uint to_uint(const string& s, const uint default_value) {
	const string t = trim(s);
	return parse_sanity_check(t, "\\+?\\d+") ? (uint)atoi(t.c_str()) : default_value;
}
inline slong to_slong(const string& s, const slong default_value) {
	const string t = trim(s);
	return parse_sanity_check(t, "[+-]?\\d+") ? (slong)atoll(t.c_str()) : default_value;
}
inline ulong to_ulong(const string& s, const ulong default_value) {
	const string t = trim(s);
	return parse_sanity_check(t, "\\+?\\d+") ? (ulong)atoll(t.c_str()) : default_value;
}
inline float to_float(const string& s, const float default_value) {
	const string t = trim(s);
	return parse_sanity_check(t, "[+-]?(((\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+[fF]?)?)|(\\d+\\.\\d*|\\.\\d+)[fF]?)") ? (float)atof(t.c_str()) : default_value;
}
inline double to_double(const string& s, const double default_value) {
	const string t = trim(s);
	return parse_sanity_check(t, "[+-]?(((\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+[fF]?)?)|(\\d+\\.\\d*|\\.\\d+)[fF]?)") ? atof(t.c_str()) : default_value;
}
#endif // UTILITIES_REGEX

#ifdef UTILITIES_FILE
#include <fstream> // read/write files
#ifndef UTILITIES_NO_CPP17
#include <filesystem> // automatically create directory before writing file, requires C++17
inline vector<string> find_files(const string& path, const string& extension=".*") {
	vector<string> files;
	if(std::filesystem::is_directory(path)&&std::filesystem::exists(path)) {
		for(const auto& entry : std::filesystem::directory_iterator(path)) {
			if(extension==".*"||entry.path().extension().string()==extension) files.push_back(entry.path().string());
		}
	}
	return files;
}
#endif // UTILITIES_NO_CPP17
inline void create_folder(const string& path) { // create folder if it not already exists
	const int slash_position = (int)path.rfind('/'); // find last slash dividing the path from the filename
	if(slash_position==(int)string::npos) return; // no slash found
	const string f = path.substr(0, slash_position); // cut off file name if there is any
#ifndef UTILITIES_NO_CPP17
	if(!std::filesystem::is_directory(f)||!std::filesystem::exists(f)) std::filesystem::create_directories(f); // create folder if it not already exists
#endif // UTILITIES_NO_CPP17
}
inline string create_file_extension(const string& path, const string& extension) {
	return path.substr(0, path.rfind('.'))+(extension.at(0)!='.'?".":"")+extension; // remove existing file extension if existing and replace it with new one
}
inline string read_file(const string& path) {
	std::ifstream file(path, std::ios::in);
	if(file.fail()) println("\rError: File \""+path+"\" does not exist!");
	const string r((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	file.close();
	return r;
}
inline void write_file(const string& path, const string& content="") {
	create_folder(path);
	std::ofstream file(path, std::ios::out);
	file.write(content.c_str(), content.length());
	file.close();
}
#endif // UTILITIES_FILE