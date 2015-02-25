using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Parse
    {
        public static int GetToken(string z, int offset, ref TK tokenType)
        {
            int i;
            char c = '\0';
            switch (z[offset + 0])
            {
                case ' ':
                case '\t':
                case '\n':
                case '\f':
                case '\r':
                    {
                        C.ASSERTCOVERAGE(z[offset + 0] == ' ');
                        C.ASSERTCOVERAGE(z[offset + 0] == '\t');
                        C.ASSERTCOVERAGE(z[offset + 0] == '\n');
                        C.ASSERTCOVERAGE(z[offset + 0] == '\f');
                        C.ASSERTCOVERAGE(z[offset + 0] == '\r');
                        for (i = 1; z.Length > offset + i && char.IsWhiteSpace(z[offset + i]); i++) { }
                        tokenType = TK.SPACE;
                        return i;
                    }
                case '-':
                    {
                        if (z.Length > offset + 1 && z[offset + 1] == '-')
                        {
                            // IMP: R-15891-05542 -- syntax diagram for comments
                            for (i = 2; z.Length > offset + i && (c = z[offset + i]) != '\0' && c != '\n'; i++) { }
                            tokenType = TK.SPACE;   // IMP: R-22934-25134
                            return i;
                        }
                        tokenType = TK.MINUS;
                        return 1;
                    }
                case '(':
                    {
                        tokenType = TK.LP;
                        return 1;
                    }
                case ')':
                    {
                        tokenType = TK.RP;
                        return 1;
                    }
                case ';':
                    {
                        tokenType = TK.SEMI;
                        return 1;
                    }
                case '+':
                    {
                        tokenType = TK.PLUS;
                        return 1;
                    }
                case '*':
                    {
                        tokenType = TK.STAR;
                        return 1;
                    }
                case '/':
                    {
                        if (offset + 2 >= z.Length || z[offset + 1] != '*')
                        {
                            tokenType = TK.SLASH;
                            return 1;
                        }
                        // IMP: R-15891-05542 -- syntax diagram for comments
                        for (i = 3, c = z[offset + 2]; offset + i < z.Length && (c != '*' || (z[offset + i] != '/') && (c != '\0')); i++) { c = z[offset + i]; }
                        if (offset + i == z.Length) c = '\0';
                        if (c != '\0') i++;
                        tokenType = TK.SPACE; // IMP: R-22934-25134
                        return i;
                    }
                case '%':
                    {
                        tokenType = TK.REM;
                        return 1;
                    }
                case '=':
                    {
                        tokenType = TK.EQ;
                        return 1 + (z[offset + 1] == '=' ? 1 : 0);
                    }
                case '<':
                    {
                        if ((c = z[offset + 1]) == '=')
                        {
                            tokenType = TK.LE;
                            return 2;
                        }
                        else if (c == '>')
                        {
                            tokenType = TK.NE;
                            return 2;
                        }
                        else if (c == '<')
                        {
                            tokenType = TK.LSHIFT;
                            return 2;
                        }
                        else
                        {
                            tokenType = TK.LT;
                            return 1;
                        }
                    }
                case '>':
                    {
                        if (z.Length > offset + 1 && (c = z[offset + 1]) == '=')
                        {
                            tokenType = TK.GE;
                            return 2;
                        }
                        else if (c == '>')
                        {
                            tokenType = TK.RSHIFT;
                            return 2;
                        }
                        else
                        {
                            tokenType = TK.GT;
                            return 1;
                        }
                    }
                case '!':
                    {
                        if (z[offset + 1] != '=')
                        {
                            tokenType = TK.ILLEGAL;
                            return 2;
                        }
                        else
                        {
                            tokenType = TK.NE;
                            return 2;
                        }
                    }
                case '|':
                    {
                        if (z[offset + 1] != '|')
                        {
                            tokenType = TK.BITOR;
                            return 1;
                        }
                        else
                        {
                            tokenType = TK.CONCAT;
                            return 2;
                        }
                    }
                case ',':
                    {
                        tokenType = TK.COMMA;
                        return 1;
                    }
                case '&':
                    {
                        tokenType = TK.BITAND;
                        return 1;
                    }
                case '~':
                    {
                        tokenType = TK.BITNOT;
                        return 1;
                    }
                case '`':
                case '\'':
                case '"':
                    {
                        int delim = z[offset + 0];
                        C.ASSERTCOVERAGE(delim == '`');
                        C.ASSERTCOVERAGE(delim == '\'');
                        C.ASSERTCOVERAGE(delim == '"');
                        for (i = 1; (offset + i) < z.Length && (c = z[offset + i]) != '\0'; i++)
                        {
                            if (c == delim)
                            {
                                if (z.Length > offset + i + 1 && z[offset + i + 1] == delim)
                                    i++;
                                else
                                    break;
                            }
                        }
                        if ((offset + i == z.Length && c != delim) || z[offset + i] != delim)
                        {
                            tokenType = TK.ILLEGAL;
                            return i + 1;
                        }
                        if (c == '\'')
                        {
                            tokenType = TK.STRING;
                            return i + 1;
                        }
                        else if (c != 0)
                        {
                            tokenType = TK.ID;
                            return i + 1;
                        }
                        else
                        {
                            tokenType = TK.ILLEGAL;
                            return i;
                        }
                    }
                case '.':
                    {
#if !OMIT_FLOATING_POINT
                        if (!char.IsDigit(z[offset + 1]))
#endif
                        {
                            tokenType = TK.DOT;
                            return 1;
                        }
                        // If the next character is a digit, this is a floating point number that begins with ".".  Fall thru into the next case
                        goto case '0';
                    }
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    {
                        C.ASSERTCOVERAGE(z[offset] == '0');
                        C.ASSERTCOVERAGE(z[offset] == '1');
                        C.ASSERTCOVERAGE(z[offset] == '2');
                        C.ASSERTCOVERAGE(z[offset] == '3');
                        C.ASSERTCOVERAGE(z[offset] == '4');
                        C.ASSERTCOVERAGE(z[offset] == '5');
                        C.ASSERTCOVERAGE(z[offset] == '6');
                        C.ASSERTCOVERAGE(z[offset] == '7');
                        C.ASSERTCOVERAGE(z[offset] == '8');
                        C.ASSERTCOVERAGE(z[offset] == '9');
                        tokenType = TK.INTEGER;
                        for (i = 0; z.Length > offset + i && char.IsDigit(z[offset + i]); i++) { }
#if !OMIT_FLOATING_POINT
                        if (z.Length > offset + i && z[offset + i] == '.')
                        {
                            i++;
                            while (z.Length > offset + i && char.IsDigit(z[offset + i])) i++;
                            tokenType = TK.FLOAT;
                        }
                        if (z.Length > offset + i + 1 && (z[offset + i] == 'e' || z[offset + i] == 'E') && (char.IsDigit(z[offset + i + 1]) || z.Length > offset + i + 2 && ((z[offset + i + 1] == '+' || z[offset + i + 1] == '-') && char.IsDigit(z[offset + i + 2]))))
                        {
                            i += 2;
                            while (z.Length > offset + i && char.IsDigit(z[offset + i])) i++;
                            tokenType = TK.FLOAT;
                        }
#endif
                        while (offset + i < z.Length && IsIdChar(z[offset + i]))
                        {
                            tokenType = TK.ILLEGAL;
                            i++;
                        }
                        return i;
                    }

                case '[':
                    {
                        for (i = 1, c = z[offset + 0]; c != ']' && (offset + i) < z.Length && (c = z[offset + i]) != '\0'; i++) { }
                        tokenType = (c == ']' ? TK.ID : TK.ILLEGAL);
                        return i;
                    }
                case '?':
                    {
                        tokenType = TK.VARIABLE;
                        for (i = 1; z.Length > offset + i && char.IsDigit(z[offset + i]); i++) { }
                        return i;
                    }
                case '#':
                    {
                        for (i = 1; z.Length > offset + i && char.IsDigit(z[offset + i]); i++) { }
                        if (i > 1)
                        {
                            // Parameters of the form #NNN (where NNN is a number) are used internally by sqlite3NestedParse.
                            tokenType = TK.REGISTER;
                            return i;
                        }
                        // Fall through into the next case if the '#' is not followed by a digit. Try to match #AAAA where AAAA is a parameter name.
                        goto case ':';
                    }
#if !OMIT_TCL_VARIABLE
                case '$':
#endif
                case '@':  // For compatibility with MS SQL Server
                case ':':
                    {
                        int n = 0;
                        C.ASSERTCOVERAGE(z[offset + 0] == '$');
                        C.ASSERTCOVERAGE(z[offset + 0] == '@');
                        C.ASSERTCOVERAGE(z[offset + 0] == ':');
                        tokenType = TK.VARIABLE;
                        for (i = 1; z.Length > offset + i && (c = z[offset + i]) != '\0'; i++)
                        {
                            if (IsIdChar(c))
                            {
                                n++;
#if !OMIT_TCL_VARIABLE
                            }
                            else if (c == '(' && n > 0)
                            {
                                do
                                {
                                    i++;
                                } while ((offset + i) < z.Length && (c = z[offset + i]) != 0 && !char.IsWhiteSpace(c) && c != ')');
                                if (c == ')')
                                    i++;
                                else
                                    tokenType = TK.ILLEGAL;
                                break;
                            }
                            else if (c == ':' && z[offset + i + 1] == ':')
                            {
                                i++;
#endif
                            }
                            else
                                break;
                        }
                        if (n == 0) tokenType = TK.ILLEGAL;
                        return i;
                    }
#if !OMIT_BLOB_LITERAL
                case 'x':
                case 'X':
                    {
                        C.ASSERTCOVERAGE(z[offset + 0] == 'x');
                        C.ASSERTCOVERAGE(z[offset + 0] == 'X');
                        if (z.Length > offset + 1 && z[offset + 1] == '\'')
                        {
                            tokenType = TK.BLOB;
                            for (i = 2; z.Length > offset + i && IsXDigit(z[offset + i]); i++) { }
                            if (offset + i == z.Length || z[offset + i] != '\'' || i % 2 != 0)
                            {
                                tokenType = TK.ILLEGAL;
                                while (z.Length > offset + i && z[offset + i] != '\'') i++;
                            }
                            if (z.Length > offset + i) i++;
                            return i;
                        }
                        goto default;
                        // Otherwise fall through to the next case
                    }
#endif
                default:
                    {
                        if (!IsIdChar(z[offset]))
                            break;
                        for (i = 1; i < z.Length - offset && IsIdChar(z[offset + i]); i++) { }
                        tokenType = KeywordCode(z, offset, i);
                        return i;
                    }
            }
            tokenType = TK.ILLEGAL;
            return 1;
        }

        private static TK KeywordCode(string z, int offset, int i)
        {
            throw new NotImplementedException();
        }

        static bool IsXDigit(char c) { return true; }
        static bool IsIdChar(char c) { return true; }

        public int RunParser(string sql, ref string errMsg)
        {
            Debug.Assert(errMsg != null);
            Debug.Assert(NewTable == null);
            Debug.Assert(NewTrigger == null);
            Debug.Assert(VarsSeen == 0);
            Debug.Assert(Vars.length == 0);
            Debug.Assert(Vars.data == null);
            Context ctx = Ctx; // The database connection
            int maxSqlLength = ctx.Limits[(int)LIMIT.SQL_LENGTH]; // Max length of an SQL string
            if (ctx.ActiveVdbeCnt == 0)
                ctx.u1.IsInterrupted = false;
            RC = RC.OK;
            Tail = new StringBuilder(sql);
            yyParser engine = Parser_Alloc(); // The LEMON-generated LALR(1) parser
            if (engine == null)
            {
                ctx.MallocFailed = true;
                return (int)RC.NOMEM;
            }
            bool enableLookaside = ctx.Lookaside.Enabled; // Saved value of db->lookaside.bEnabled
            if (ctx.Lookaside.Start != 0)
                ctx.Lookaside.Enabled = true;
            int errs = 0;                   // Number of errors encountered
            TK tokenType = 0;              // type of the next token
            TK lastTokenParsed = 0;       // type of the previous token
            int i = 0;
            while (/*  !ctx.MallocFailed && */ i < sql.Length)
            {
                Debug.Assert(i >= 0);
                LastToken.data = sql.Substring(i);
                LastToken.length = (uint)GetToken(sql, i, ref tokenType);
                i += (int)LastToken.length;
                if (i > maxSqlLength)
                {
                    RC = RC.TOOBIG;
                    break;
                }
                switch (tokenType)
                {
                    case TK.SPACE:
                        {
                            if (ctx.u1.IsInterrupted)
                            {
                                ErrorMsg("interrupt");
                                RC = RC.INTERRUPT;
                                goto abort_parse;
                            }
                            break;
                        }
                    case TK.ILLEGAL:
                        {
                            C._tagfree(ctx, ref errMsg);
                            errMsg = SysEx.Mprintf(ctx, "unrecognized token: \"%T\"", (object)LastToken);
                            errs++;
                            goto abort_parse;
                        }
                    case TK.SEMI:
                        {
                            //Tail = new StringBuilder(sql.Substring(i, sql.Length - i));
                            goto default;
                        }
                    default:
                        {
                            Parser(engine, tokenType, LastToken, this);
                            lastTokenParsed = tokenType;
                            if (RC != RC.OK)
                                goto abort_parse;
                            break;
                        }
                }
            }
        abort_parse:
            Tail = new StringBuilder(sql.Length <= i ? string.Empty : sql.Substring(i, sql.Length - i));
            if (sql.Length >= i && errs == 0 && RC == RC.OK)
            {
                if (lastTokenParsed != TK.SEMI)
                    Parser(engine, TK.SEMI, LastToken, this);
                Parser(engine, 0, LastToken, this);
            }
#if YYTRACKMAXSTACKDEPTH
            sqlite3StatusSet(SQLITE_STATUS_PARSER_STACK, sqlite3ParserStackPeak(engine));
#endif
            Parser_Free(engine, null);
            ctx.Lookaside.Enabled = enableLookaside;
            //if (ctx.MallocFailed)
            //    RC = RC.NOMEM;
            if (RC != RC.OK && RC != RC.DONE && string.IsNullOrEmpty(ErrMsg))
                SetString(ref ErrMsg, ctx, ErrStr(RC));
            if (ErrMsg != null)
            {
                errMsg = ErrMsg;
                SysEx.LOG(RC, "%s", errMsg);
                ErrMsg = string.Empty;
                errs++;
            }
            if (V != null && Errs > 0 && Nested == 0)
            {
                Vdbe.Delete(ref V);
                V = null;
            }
#if !OMIT_SHARED_CACHE
            if (Nested == 0)
            {
                C._tagfree(ctx, ref TableLocks.data);
                TableLocks.data = null;
                TableLocks.length = 0;
            }
#endif
#if !OMIT_VIRTUALTABLE
            VTableLocks.data = null;
#endif
            if (!E.INDECLARE_VTABLE(this))
            {
                // If the pParse.declareVtab flag is set, do not delete any table structure built up in pParse.pNewTable. The calling code (see vtab.c)
                // will take responsibility for freeing the Table structure.
                DeleteTable(ctx, ref NewTable);
            }

#if !OMIT_TRIGGER
            DeleteTrigger(ctx, ref NewTrigger);
#endif
            //for (i = Vars.length - 1; i >= 0; i--)
            //    C._tagfree(ctx, ref Vars.data[i]);
            C._tagfree(ctx, ref Vars.data);
            C._tagfree(ctx, ref Alias.data);
            while (Ainc != null)
            {
                AutoincInfo p = Ainc;
                Ainc = p.Next;
                C._tagfree(ctx, ref p);
            }
            while (ZombieTab != null)
            {
                Table p = ZombieTab;
                ZombieTab = p.NextZombie;
                DeleteTable(ctx, ref p);
            }
            if (errs > 0 && RC == RC.OK)
                RC = RC.ERROR;
            return errs;
        }

        public static int Dequote(ref string z)
        {
            if (string.IsNullOrEmpty(z)) return -1;
            char quote = z[0];
            switch (quote)
            {
                case '\'': break;
                case '"': break;
                case '`': break;                // For MySQL compatibility
                case '[': quote = ']'; break;   // For MS SqlServer compatibility
                default: return -1;
            }
            StringBuilder b = new StringBuilder(z.Length);
            int i;
            for (i = 1; i < z.Length; i++)
                if (z[i] == quote)
                {
                    if (i < z.Length - 1 && (z[i + 1] == quote)) { b.Append(quote); i++; }
                    else break;
                }
                else
                    b.Append(z[i]);
            z = b.ToString();
            return b.Length;
        }
    }
}
