// tokenize.c
#include "Core+Vdbe.cu.h"
#include "../KeywordHash.h"

namespace Core
{
	__device__ int Parse::GetToken(const unsigned char *z, int *tokenType)
	{
		int i, c;
		switch (*z)
		{
		case ' ': case '\t': case '\n': case '\f': case '\r': {
			ASSERTCOVERAGE(z[0] == ' ');
			ASSERTCOVERAGE(z[0] == '\t');
			ASSERTCOVERAGE(z[0] == '\n');
			ASSERTCOVERAGE(z[0] == '\f');
			ASSERTCOVERAGE(z[0] == '\r');
			for (i = 1; _isspace(z[i]); i++) { }
			*tokenType = TK_SPACE;
			return i; }
		case '-': {
			if (z[1] == '-')
			{
				// IMP: R-50417-27976 -- syntax diagram for comments
				for (i = 2; (c = z[i]) != 0 && c != '\n'; i++) { }
				*tokenType = TK_SPACE; // IMP: R-22934-25134
				return i;
			}
			*tokenType = TK_MINUS;
			return 1; }
		case '(': {
			*tokenType = TK_LP;
			return 1; }
		case ')': {
			*tokenType = TK_RP;
			return 1; }
		case ';': {
			*tokenType = TK_SEMI;
			return 1; }
		case '+': {
			*tokenType = TK_PLUS;
			return 1; }
		case '*': {
			*tokenType = TK_STAR;
			return 1; }
		case '/': {
			if (z[1] != '*' || z[2] == 0)
			{
				*tokenType = TK_SLASH;
				return 1;
			}
			// IMP: R-50417-27976 -- syntax diagram for comments
			for (i = 3, c = z[2]; (c != '*' || z[i] != '/') && (c = z[i]) != 0; i++) { }
			if (c) i++;
			*tokenType = TK_SPACE; // IMP: R-22934-25134
			return i; }
		case '%': {
			*tokenType = TK_REM;
			return 1; }
		case '=': {
			*tokenType = TK_EQ;
			return 1 + (z[1] == '='); }
		case '<': {
			if ((c = z[1]) == '=')
			{
				*tokenType = TK_LE;
				return 2;
			}
			else if (c == '>')
			{
				*tokenType = TK_NE;
				return 2;
			}
			else if (c == '<')
			{
				*tokenType = TK_LSHIFT;
				return 2;
			}
			else
			{
				*tokenType = TK_LT;
				return 1;
			} }
		case '>': {
			if ((c = z[1]) == '=')
			{
				*tokenType = TK_GE;
				return 2;
			}
			else if (c == '>')
			{
				*tokenType = TK_RSHIFT;
				return 2;
			}
			else
			{
				*tokenType = TK_GT;
				return 1;
			} }
		case '!': {
			if (z[1] != '=')
			{
				*tokenType = TK_ILLEGAL;
				return 2;
			}
			else
			{
				*tokenType = TK_NE;
				return 2;
			} }
		case '|': {
			if (z[1] != '|')
			{
				*tokenType = TK_BITOR;
				return 1;
			}
			else
			{
				*tokenType = TK_CONCAT;
				return 2;
			} }
		case ',': {
			*tokenType = TK_COMMA;
			return 1; }
		case '&': {
			*tokenType = TK_BITAND;
			return 1; }
		case '~': {
			*tokenType = TK_BITNOT;
			return 1; }
		case '`':
		case '\'':
		case '"': {
			int delim = z[0];
			ASSERTCOVERAGE(delim == '`');
			ASSERTCOVERAGE(delim == '\'');
			ASSERTCOVERAGE(delim == '"');
			for (i = 1; (c = z[i]) != 0; i++)
			{
				if (c == delim)
				{
					if (z[i+1] == delim)
						i++;
					else
						break;
				}
			}
			if (c == '\'')
			{
				*tokenType = TK_STRING;
				return i+1;
			}
			else if (c != 0)
			{
				*tokenType = TK_ID;
				return i+1;
			}
			else
			{
				*tokenType = TK_ILLEGAL;
				return i;
			} }
		case '.': {
#ifndef OMIT_FLOATING_POINT
			if (!_isdigit(z[1]))
#endif
			{
				*tokenType = TK_DOT;
				return 1;
			} }
				  // If the next character is a digit, this is a floating point number that begins with ".".  Fall thru into the next case
		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9': {
			ASSERTCOVERAGE(z[0] == '0'); ASSERTCOVERAGE(z[0] == '1'); ASSERTCOVERAGE(z[0] == '2');
			ASSERTCOVERAGE(z[0] == '3'); ASSERTCOVERAGE(z[0] == '4'); ASSERTCOVERAGE(z[0] == '5');
			ASSERTCOVERAGE(z[0] == '6'); ASSERTCOVERAGE(z[0] == '7'); ASSERTCOVERAGE(z[0] == '8');
			ASSERTCOVERAGE(z[0] == '9');
			*tokenType = TK_INTEGER;
			for (i = 0; _isdigit(z[i]); i++) { }
#ifndef OMIT_FLOATING_POINT
			if (z[i] == '.')
			{
				i++;
				while (_isdigit(z[i])) i++;
				*tokenType = TK_FLOAT;
			}
			if ((z[i] == 'e' || z[i] == 'E') && _isdigit(z[i+1] || ((z[i+1] == '+' || z[i+1] == '-') && _isdigit(z[i+2]))))
			{
				i += 2;
				while (_isdigit(z[i])) i++;
				*tokenType = TK_FLOAT;
			}
#endif
			while (_isidchar(z[i]))
			{
				*tokenType = TK_ILLEGAL;
				i++;
			}
			return i; }
		case '[': {
			for (i = 1, c = z[0]; c != ']' && (c = z[i]) != 0; i++) { }
			*tokenType = (c == ']' ? TK_ID : TK_ILLEGAL);
			return i; }
		case '?': {
			*tokenType = TK_VARIABLE;
			for (i = 1; _isdigit(z[i]); i++) { }
			return i; }
		case '#': {
			for (i = 1; _isdigit(z[i]); i++) { }
			if (i > 1)
			{
				// Parameters of the form #NNN (where NNN is a number) are used internally by sqlite3NestedParse.
				*tokenType = TK_REGISTER;
				return i;
			} }
				  // Fall through into the next case if the '#' is not followed by a digit. Try to match #AAAA where AAAA is a parameter name.
#ifndef TCL_VARIABLE
		case '$':
#endif
		case '@':  // For compatibility with MS SQL Server
		case ':': {
			int n = 0;
			ASSERTCOVERAGE(z[0] == '$'); ASSERTCOVERAGE(z[0] == '@'); ASSERTCOVERAGE(z[0] == ':');
			*tokenType = TK_VARIABLE;
			for (i = 1; (c = z[i]) != 0; i++)
			{
				if (_isidchar(c))
				{
					n++;
#ifndef OMIT_TCL_VARIABLE
				}
				else if (c == '(' && n > 0)
				{
					do
					{
						i++;
					} while ((c = z[i]) != 0 && !_isspace(c) && c != ')');
					if (c == ')')
						i++;
					else
						*tokenType = TK_ILLEGAL;
					break;
				}
				else if (c == ':' && z[i+1] == ':')
				{
					i++;
#endif
				}
				else
					break;
			}
			if (n == 0) *tokenType = TK_ILLEGAL;
			return i; }
#ifndef OMIT_BLOB_LITERAL
		case 'x': case 'X': {
			ASSERTCOVERAGE(z[0] == 'x'); ASSERTCOVERAGE(z[0] == 'X');
			if (z[1] == '\'')
			{
				*tokenType = TK_BLOB;
				for (i = 2; _isxdigit(z[i]); i++) { }
				if (z[i] != '\'' || i % 2)
				{
					*tokenType = TK_ILLEGAL;
					while (z[i] && z[i] != '\'') i++;
				}
				if (z[i]) i++;
				return i;
			} }
				  // Otherwise fall through to the next case
#endif
		default: {
			if (!_isidchar(*z))
				break;
			for (i = 1; _isidchar(z[i]); i++) { }
			*tokenType = KeywordCode((char *)z, i);
			return i; }
		}
		*tokenType = TK_ILLEGAL;
		return 1;
	}

	__device__ int Parse::RunParser(const char *sql, char **errMsg)
	{
		_assert(errMsg);
		_assert(!NewTable);
		_assert(!NewTrigger);
		_assert(VarsSeen == 0);
		_assert(Vars.length == 0);
		_assert(!Vars.data);
		Context *ctx = Ctx;       // The database connection
		int maxSqlLength = ctx->Limits[LIMIT_SQL_LENGTH]; // Max length of an SQL string
		if (ctx->ActiveVdbeCnt == 0)
			ctx->u1.IsInterrupted = false;
		RC = RC_OK;
		Tail = sql;
		void *engine = ParserAlloc((void *(*)(size_t))_alloc); // The LEMON-generated LALR(1) parser
		if (!engine)
		{
			ctx->MallocFailed = true;
			return RC_NOMEM;
		}
		bool enableLookaside = ctx->Lookaside.Enabled; // Saved value of db->lookaside.bEnabled
		if (ctx->Lookaside.Start)
			ctx->Lookaside.Enabled = true;
		int errs = 0; // Number of errors encountered
		int tokenType;                  // type of the next token
		int lastTokenParsed = 0;       // type of the previous token
		int i = 0;
		while (!ctx->MallocFailed && sql[i] != 0)
		{
			_assert(i >= 0);
			LastToken.data = (char *)&sql[i];
			LastToken.length = GetToken((unsigned char *)&sql[i], &tokenType);
			i += LastToken.length;
			if (i > maxSqlLength)
			{
				RC = RC_TOOBIG;
				break;
			}
			switch (tokenType)
			{
			case TK_SPACE: {
				if (ctx->u1.IsInterrupted)
				{
					ErrorMsg("interrupt");
					RC = RC_INTERRUPT;
					goto abort_parse;
				}
				break; }
			case TK_ILLEGAL: {
				_tagfree(ctx, *errMsg);
				*errMsg = _mtagprintf(ctx, "unrecognized token: \"%T\"", &LastToken);
				errs++;
				goto abort_parse; }
			case TK_SEMI: {
				Tail = &sql[i]; }
						  // Fall thru into the default case
			default: {
				Parser(engine, tokenType, LastToken, this);
				lastTokenParsed = tokenType;
				if (RC != RC_OK)
					goto abort_parse;
				break; }
			}
		}
abort_parse:
		if (sql[i] == 0 && errs == 0 && RC == RC_OK)
		{
			if (lastTokenParsed != TK_SEMI)
			{
				Parser(engine, TK_SEMI, LastToken, this);
				Tail = &sql[i];
			}
			Parser(engine, 0, LastToken, this);
		}
#ifdef YYTRACKMAXSTACKDEPTH
		StatusEx::Set(StatusEx::STATUS_PARSER_STACK, ParserStackPeak(engine));
#endif
		ParserFree(engine, _free);
		ctx->Lookaside.Enabled = enableLookaside;
		if (ctx->MallocFailed)
			RC = RC_NOMEM;
		if (RC != RC_OK && RC != RC_DONE && !ErrMsg)
			_setstring(&ErrMsg, ctx, "%s", Main::ErrStr(RC));
		_assert(errMsg);
		if (ErrMsg)
		{
			*errMsg = ErrMsg;
			SysEx_LOG(RC, "%s", *errMsg);
			ErrMsg = nullptr;
			errs++;
		}
		if (V && Errs > 0 && Nested == 0)
		{
			Vdbe::Delete(V);
			V = nullptr;
		}
#ifndef OMIT_SHARED_CACHE
		if (Nested == 0)
		{
			_tagfree(ctx, TableLocks.data);
			TableLocks.data = nullptr;
			TableLocks.length = 0;
		}
#endif
#ifndef OMIT_VIRTUALTABLE
		_free(VTableLocks.data);
#endif

		if (!INDECLARE_VTABLE(this))
		{
			// If the pParse->declareVtab flag is set, do not delete any table structure built up in pParse->pNewTable. The calling code (see vtab.c)
			// will take responsibility for freeing the Table structure.
			DeleteTable(ctx, NewTable);
		}
#if !OMIT_TRIGGER
		Trigger::DeleteTrigger(ctx, NewTrigger);
#endif
		for (i = Vars.length-1; i >= 0; i--)
			_tagfree(ctx, Vars.data[i]);
		_tagfree(ctx, Vars.data);
		_tagfree(ctx, Alias.data);
		while (Ainc)
		{
			AutoincInfo *p = Ainc;
			Ainc = p->Next;
			_tagfree(ctx, p);
		}
		while (ZombieTab)
		{
			Table *p = ZombieTab;
			ZombieTab = p->NextZombie;
			DeleteTable(ctx, p);
		}
		if (errs > 0 && RC == RC_OK)
			RC = RC_ERROR;
		return errs;
	}

	__device__ int Parse::Dequote(char *z)
	{
		if (!z) return -1;
		char quote = z[0];
		switch (quote)
		{
		case '\'': break;
		case '"': break;
		case '`': break;                // For MySQL compatibility
		case '[': quote = ']'; break;  // For MS SqlServer compatibility
		default: return -1;
		}
		int i, j;
		for (i = 1, j = 0; _ALWAYS(z[i]); i++)
			if (z[i] == quote)
			{
				if (z[i+1] == quote) { z[j++] = quote; i++; }
				else break;
			}
			else
				z[j++] = z[i];
		z[j] = 0;
		return j;
	}
}
