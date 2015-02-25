// complete.c
#region OMIT_COMPLETE
#if !OMIT_COMPLETE
using System;
using System.Diagnostics;

namespace Core
{
    public partial class Parser
    {

        enum TKC : byte
        {
            SEMI = 0,
            WS = 1,
            OTHER = 2,
#if !OMIT_TRIGGER
            EXPLAIN = 3,
            CREATE = 4,
            TEMP = 5,
            TRIGGER = 6,
            END = 7,
#endif
        }

#if !OMIT_TRIGGER
        static byte[][] _trans = new byte[][]
      {
     /* State:       **  SEMI  WS  OTHER  EXPLAIN  CREATE  TEMP  TRIGGER  END */
     /* 0 INVALID: */ new byte[]{    1,  0,     2,       3,      4,    2,       2,   2, },
     /* 1   START: */ new byte[]{    1,  1,     2,       3,      4,    2,       2,   2, },
     /* 2  NORMAL: */ new byte[]{    1,  2,     2,       2,      2,    2,       2,   2, },
     /* 3 EXPLAIN: */ new byte[]{    1,  3,     3,       2,      4,    2,       2,   2, },
     /* 4  CREATE: */ new byte[]{    1,  4,     2,       2,      2,    4,       5,   2, },
     /* 5 TRIGGER: */ new byte[]{    6,  5,     5,       5,      5,    5,       5,   5, },
     /* 6    SEMI: */ new byte[]{    6,  6,     5,       5,      5,    5,       5,   7, },
     /* 7     END: */ new byte[]{    1,  7,     5,       5,      5,    5,       5,   5, } };
#else
      static byte[][] _trans = new byte[][] {
     /* State:       **  SEMI  WS  OTHER */
     /* 0 INVALID: */new byte[] {    1,  0,     2, },
     /* 1   START: */new byte[] {    1,  1,     2, },
     /* 2  NORMAL: */new byte[] {    1,  2,     2, } };
#endif

        public static bool Complete(string sql)
        {
            int state = 0; // Current state, using numbers defined in header comment
            TKC token; // Value of the next token
            int sqlIdx = 0;
            while (sqlIdx < sql.Length)
            {
                switch (sql[sqlIdx])
                {
                    case ';':
                        { // A semicolon
                            token = TKC.SEMI;
                            break;
                        }

                    case ' ':
                    case '\r':
                    case '\t':
                    case '\n':
                    case '\f':
                        { // White space is ignored
                            token = TKC.WS;
                            break;
                        }

                    case '/':
                        {   // C-style comments
                            if (sql[sqlIdx + 1] != '*')
                            {
                                token = TKC.OTHER;
                                break;
                            }
                            sqlIdx += 2;
                            while (sqlIdx < sql.Length && sql[sqlIdx] != '*' || sqlIdx < sql.Length - 1 && sql[sqlIdx + 1] != '/') sqlIdx++;
                            if (sqlIdx == sql.Length) return false;
                            sqlIdx++;
                            token = TKC.WS;
                            break;
                        }

                    case '-':
                        { // SQL-style comments from "--" to end of line
                            if (sql[sqlIdx + 1] != '-')
                            {
                                token = TKC.OTHER;
                                break;
                            }
                            while (sqlIdx < sql.Length && sql[sqlIdx] != '\n') sqlIdx++;
                            if (sqlIdx == sql.Length) return (state == 1);
                            token = TKC.WS;
                            break;
                        }

                    case '[':
                        { // Microsoft-style identifiers in [...]
                            sqlIdx++;
                            while (sqlIdx < sql.Length && sql[sqlIdx] != ']') sqlIdx++;
                            if (sqlIdx == sql.Length) return false;
                            token = TKC.OTHER;
                            break;
                        }

                    case '`': // Grave-accent quoted symbols used by MySQL
                    case '"': // single- and double-quoted strings
                    case '\'':
                        {
                            int c = sql[sqlIdx];
                            sqlIdx++;
                            while (sqlIdx < sql.Length && sql[sqlIdx] != c) sqlIdx++;
                            if (sqlIdx == sql.Length) return false;
                            token = TKC.OTHER;
                            break;
                        }
                    default:
                        {
                            if (char.IsIdChar(sql[sqlIdx]))
                            {
                                // Keywords and unquoted identifiers
                                int id;
                                for (id = 1; (sqlIdx + id) < sql.Length && char.IdChar(sql[sqlIdx + id]); id++) { }
#if OMIT_TRIGGER
                                token = TKC.OTHER;
#else
                                switch (sql[sqlIdx])
                                {
                                    case 'c':
                                    case 'C':
                                        {
                                            if (id == 6 && string.Compare(sql, sqlIdx, "create", 0, 6, StringComparison.OrdinalIgnoreCase) == 0) token = TKC.CREATE;
                                            else token = TKC.OTHER;
                                            break;
                                        }

                                    case 't':
                                    case 'T':
                                        {
                                            if (id == 7 && string.Compare(sql, sqlIdx, "trigger", 0, 7, StringComparison.OrdinalIgnoreCase) == 0) token = TKC.TRIGGER;
                                            else if (id == 4 && string.Compare(sql, sqlIdx, "temp", 0, 4, StringComparison.OrdinalIgnoreCase) == 0) token = TKC.TEMP;
                                            else if (id == 9 && string.Compare(sql, sqlIdx, "temporary", 0, 9, StringComparison.OrdinalIgnoreCase) == 0) token = TKC.TEMP;
                                            else token = TKC.OTHER;
                                            break;
                                        }

                                    case 'e':
                                    case 'E':
                                        {
                                            if (id == 3 && string.Compare(sql, sqlIdx, "end", 0, 3, StringComparison.OrdinalIgnoreCase) == 0) token = TKC.END;
                                            else
#if !OMIT_EXPLAIN
                                                if (id == 7 && string.Compare(sql, sqlIdx, "explain", 0, 7, StringComparison.OrdinalIgnoreCase) == 0) token = TKC.EXPLAIN;
                                                else
#endif
                                                    token = TKC.OTHER;
                                            break;
                                        }
                                    default:
                                        {
                                            token = TKC.OTHER;
                                            break;
                                        }
                                }
#endif
                                sqlIdx += id - 1;
                            }
                            else token = TKC.OTHER; // Operators and special symbols
                            break;
                        }
                }
                state = _trans[state][(int)token];
                sqlIdx++;
            }
            return (state == 1);
        }

#if !OMIT_UTF16
        public static bool Complete16(string sql)
        {
            RC rc = RC.NOMEM;
#if !OMIT_AUTOINIT
            rc = SysEx.Initialize();
            if (rc != RC.OK) return rc;
#endif
            Mem val = sqlite3ValueNew(0);
            sqlite3ValueSetStr(val, -1, sql, TEXTENCODE.UTF16NATIVE, DESTRUCTOR.STATIC);
            string sql8 = sqlite3ValueText(val, TEXTENCODE.UTF8);
            rc = (sql8 != null ? (RC)Complete(sql8) : RC.NOMEM);
            else rc = RC.NOMEM;
            sqlite3ValueFree(val);
            return Context.ApiExit(null, rc);
        }
#endif

    }
}
#endif
#endregion
