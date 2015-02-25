// complete.c
#pragma region OMIT_COMPLETE
#ifndef OMIT_COMPLETE
#include "Core+Vdbe.cu.h"

namespace Core
{
	enum TKC : uint8
	{
		TKC_SEMI = 0,
		TKC_WS = 1,
		TKC_OTHER = 2,
#ifndef OMIT_TRIGGER
		TKC_EXPLAIN = 3,
		TKC_CREATE = 4,
		TKC_TEMP = 5,
		TKC_TRIGGER = 6,
		TKC_END = 7,
#endif
	};

#ifndef OMIT_TRIGGER

	__constant__ static const uint8 _trans[8][8] = {
		/* State:       **  SEMI  WS  OTHER  EXPLAIN  CREATE  TEMP  TRIGGER  END */
		/* 0 INVALID: */ {    1,  0,     2,       3,      4,    2,       2,   2, },
		/* 1   START: */ {    1,  1,     2,       3,      4,    2,       2,   2, },
		/* 2  NORMAL: */ {    1,  2,     2,       2,      2,    2,       2,   2, },
		/* 3 EXPLAIN: */ {    1,  3,     3,       2,      4,    2,       2,   2, },
		/* 4  CREATE: */ {    1,  4,     2,       2,      2,    4,       5,   2, },
		/* 5 TRIGGER: */ {    6,  5,     5,       5,      5,    5,       5,   5, },
		/* 6    SEMI: */ {    6,  6,     5,       5,      5,    5,       5,   7, },
		/* 7     END: */ {    1,  7,     5,       5,      5,    5,       5,   5, } };
#else
	__constant__ static const uint8 _trans[3][3] = {
		/* State:       **  SEMI  WS  OTHER */
		/* 0 INVALID: */ {    1,  0,     2, },
		/* 1   START: */ {    1,  1,     2, },
		/* 2  NORMAL: */ {    1,  2,     2, } };
#endif
	__device__ bool Parse::Complete(const char *sql)
	{
		uint8 state = 0; // Current state, using numbers defined in header comment
		TKC token; // Value of the next token
		while (*sql)
		{
			switch (*sql)
			{
			case ';': { // A semicolon
				token = TKC_SEMI;
				break; }

			case ' ':
			case '\r':
			case '\t':
			case '\n':
			case '\f': { // White space is ignored
				token = TKC_WS;
				break; }

			case '/': { // C-style comments
				if (sql[1] != '*')
				{
					token = TKC_OTHER;
					break;
				}
				sql += 2;
				while (sql[0] && (sql[0] != '*' || sql[1] != '/')) sql++;
				if (sql[0] == 0) return false;
				sql++;
				token = TKC_WS;
				break; }

			case '-': { // SQL-style comments from "--" to end of line
				if (sql[1] != '-')
				{
					token = TKC_OTHER;
					break;
				}
				while (*sql && *sql != '\n') sql++;
				if (*sql == 0) return (state == 1);
				token = TKC_WS;
				break; }

			case '[': { // Microsoft-style identifiers in [...]
				sql++;
				while (*sql && *sql != ']') sql++;
				if (*sql == 0) return false;
				token = TKC_OTHER;
				break; }

			case '`': // Grave-accent quoted symbols used by MySQL
			case '"': // single- and double-quoted strings
			case '\'': {
				int c = *sql;
				sql++;
				while (*sql && *sql != c) sql++;
				if (*sql == 0) return false;
				token = TKC_OTHER;
				break; }

			default: {
				if (_isidchar(*sql))
				{
					// Keywords and unquoted identifiers
					int id;
					for (id = 1; _isidchar(sql[id]); id++) { }
#ifdef OMIT_TRIGGER
					token = TKC_OTHER;
#else
					switch (*sql)
					{
					case 'c': case 'C': {
						if (id == 6 && !_strncmp(sql, "create", 6)) token =  TKC_CREATE;
						else token = TKC_OTHER;
						break; }

					case 't': case 'T': {
						if (id == 7 && !_strncmp(sql, "trigger", 7)) token = TKC_TRIGGER;
						else if (id == 4 && !_strncmp(sql, "temp", 4)) token = TKC_TEMP;
						else if (id == 9 && !_strncmp(sql, "temporary", 9)) token = TKC_TEMP;
						else token = TKC_OTHER;
						break; }
					case 'e':  case 'E': {
						if (id == 3 && !_strncmp(sql, "end", 3)) token = TKC_END;
						else
#ifndef OMIT_EXPLAIN
							if (id == 7 && !_strncmp(sql, "explain", 7)) token = TKC_EXPLAIN;
							else
#endif 
								token = TKC_OTHER;
						break; }
					default: {
						token = TKC_OTHER;
						break; }
					}
#endif
					sql += id-1;
				}
				else token = TKC_OTHER;// Operators and special symbols
				break; }
			}
			state = _trans[state][token];
			sql++;
		}
		return (state == 1);
	}

#ifndef OMIT_UTF16
	__device__ bool Parse::Complete16(const void *sql)
	{
		::RC rc = RC_NOMEM;
#ifndef OMIT_AUTOINIT
		rc = Main::Initialize();
		if (rc) return (rc != 0);
#endif
		Mem *val = Vdbe::ValueNew(0);
		Vdbe::ValueSetStr(val, -1, sql, TEXTENCODE_UTF16NATIVE, DESTRUCTOR_STATIC);
		char const *sql8 = (const char *)Vdbe::ValueText(val, TEXTENCODE_UTF8);
		rc = (sql8 ? (::RC)Parse::Complete(sql8) : RC_NOMEM);
		Vdbe::ValueFree(val);
		return (Main::ApiExit(nullptr, rc) != 0);
	}
#endif

}
#endif
#pragma endregion
