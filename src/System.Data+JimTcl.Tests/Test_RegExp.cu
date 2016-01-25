// The code in this file implements a compact but reasonably efficient regular-expression matcher for posix extended regular
// expressions against UTF8 text.  The following syntax is supported:
//     X*      zero or more occurrences of X
//     X+      one or more occurrences of X
//     X?      zero or one occurrences of X
//     X{p,q}  between p and q occurrences of X
//     (X)     match X
//     X|Y     X or Y
//     ^X      X occurring at the beginning of the string
//     X$      X occurring at the end of the string
//     .       Match any single character
//     \c      Character c where c is one of \{}()[]|*+?.
//     \c      C-language escapes for c in afnrtv.  ex: \t or \n
//     \uXXXX  Where XXXX is exactly 4 hex digits, unicode value XXXX
//     \xXX    Where XX is exactly 2 hex digits, unicode value XX
//     [abc]   Any single character from the set abc
//     [^abc]  Any single character not in the set abc
//     [a-z]   Any single character in the range a-z
//     [^a-z]  Any single character not in the range a-z
//     \b      Word boundary
//     \w      Word character.  [A-Za-z0-9_]
//     \W      Non-word character
//     \d      Digit
//     \D      Non-digit
//     \s      Whitespace character
//     \S      Non-whitespace character
//
// A nondeterministic finite automaton (NFA) is used for matching, so the performance is bounded by O(N*M) where N is the size of the regular
// expression and M is the size of the input string.  The matcher never exhibits exponential behavior.  Note that the X{p,q} operator expands
// to p copies of X following by q-p copies of X? and that the size of the regular expression in the O(N*M) performance bound is computed after this expansion.
#include <Core+Vdbe\VdbeInt.cu.h>

// The end-of-input character
#define RE_EOF            0    // End of input

// The NFA is implemented as sequence of opcodes taken from the following set.  Each opcode has a single integer argument.
#define RE_OP_MATCH       1    // Match the one character in the argument
#define RE_OP_ANY         2    // Match any one character.  (Implements ".")
#define RE_OP_ANYSTAR     3    // Special optimized version of .*
#define RE_OP_FORK        4    // Continue to both next and opcode at iArg
#define RE_OP_GOTO        5    // Jump to opcode at iArg
#define RE_OP_ACCEPT      6    // Halt and indicate a successful match
#define RE_OP_CC_INC      7    // Beginning of a [...] character class
#define RE_OP_CC_EXC      8    // Beginning of a [^...] character class
#define RE_OP_CC_VALUE    9    // Single value in a character class
#define RE_OP_CC_RANGE   10    // Range of values in a character class
#define RE_OP_WORD       11    // Perl word character [A-Za-z0-9_]
#define RE_OP_NOTWORD    12    // Not a perl word character
#define RE_OP_DIGIT      13    // digit:  [0-9]
#define RE_OP_NOTDIGIT   14    // Not a digit
#define RE_OP_SPACE      15    // space:  [ \t\n\r\v\f]
#define RE_OP_NOTSPACE   16    // Not a digit
#define RE_OP_BOUNDARY   17    // Boundary between word and non-word

// Each opcode is a "state" in the NFA
typedef unsigned short ReStateNumber;

// Because this is an NFA and not a DFA, multiple states can be active at once.  An instance of the following object records all active states in
// the NFA.  The implementation is optimized for the common case where the number of actives states is small.
typedef array_t<ReStateNumber> ReStateSet; // Current states

// An input string read one character at a time.
typedef struct ReInput ReInput;
struct ReInput
{
	const unsigned char *z;  // All text
	int i;                   // Next byte to read
	int mx;                  // EOF when i>=mx
};

// A compiled NFA (or an NFA that is in the process of being compiled) is an instance of the following object.
typedef struct ReCompiled ReCompiled;
struct ReCompiled
{
	ReInput In;					// Regular expression text
	const char *Err;			// Error message to return
	char *OPs;                  // Operators for the virtual machine
	int *Args;                  // Arguments to each operator
	unsigned (*NextChar)(ReInput*);  // Next character function
	unsigned char Init[12];		// Initial text to match
	int InitLength;				// Number of characters in zInit
	unsigned States;			// Number of entries in aOp[] and aArg[]
	unsigned Allocs;			// Slots allocated for aOp[] and aArg[]
};

// Add a state to the given state set if it is not already there
__device__ static void re_add_state(ReStateSet *set, int newState)
{
	for (int i = 0; i < set->length; i++) if (set->data[i] == newState) return;
	set->data[set->length++] = newState;
}

// Extract the next unicode character from *pzIn and return it.  Advance *pzIn to the first byte past the end of the character returned.  To
// be clear:  this routine converts utf8 to unicode.  This routine is optimized for the common case where the next character is a single byte.
__device__ static unsigned re_next_char(ReInput *p)
{
	if (p->i >= p->mx) return 0;
	unsigned c = p->z[p->i++];
	if (c >= 0x80)
	{
		if ((c&0xe0) == 0xc0 && p->i < p->mx && (p->z[p->i]&0xc0) == 0x80)
		{
			c = (c&0x1f)<<6 | (p->z[p->i++]&0x3f);
			if (c < 0x80) c = 0xfffd;
		}
		else if ((c&0xf0) == 0xe0 && p->i+1 < p->mx && (p->z[p->i]&0xc0) == 0x80 && (p->z[p->i+1]&0xc0) == 0x80)
		{
			c = (c&0x0f)<<12 | ((p->z[p->i]&0x3f)<<6) | (p->z[p->i+1]&0x3f);
			p->i += 2;
			if (c <= 0x3ff || (c >= 0xd800 && c <= 0xdfff)) c = 0xfffd;
		}
		else if ((c&0xf8) == 0xf0 && p->i+3 < p->mx && (p->z[p->i]&0xc0) == 0x80 && (p->z[p->i+1]&0xc0) == 0x80 && (p->z[p->i+2]&0xc0) == 0x80)
		{
			c = (c&0x07)<<18 | ((p->z[p->i]&0x3f)<<12) | ((p->z[p->i+1]&0x3f)<<6) | (p->z[p->i+2]&0x3f);
			p->i += 3;
			if (c <= 0xffff || c > 0x10ffff) c = 0xfffd;
		}
		else
			c = 0xfffd;
	}
	return c;
}
__device__ static unsigned re_next_char_nocase(ReInput *p)
{
	unsigned c = re_next_char(p);
	if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
	return c;
}

// Return true if c is a perl "word" character:  [A-Za-z0-9_]
__device__ static int re_word_char(int c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }

// Return true if c is a "digit" character:  [0-9]
__device__ static int re_digit_char(int c) { return (c >= '0' && c <= '9'); }

// Return true if c is a perl "space" character:  [ \t\r\n\v\f]
__device__ static int re_space_char(int c) { return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'); }

// Run a compiled regular expression on the zero-terminated input string zIn[].  Return true on a match and false if there is no match.
__device__ int re_match(ReCompiled *re, const unsigned char *in_, int inLength)
{
	ReInput in;
	in.z = in_;
	in.i = 0;
	in.mx = (inLength >= 0 ? inLength : (int)_strlen((char const *)in_));
	// Look for the initial prefix match, if there is one.
	if (re->InitLength)
	{
		unsigned char x = re->Init[0];
		while (in.i+re->InitLength <= in.mx && (in_[in.i] != x || _strncmp((const char *)in_+in.i, (const char*)re->Init, re->InitLength) != 0))
			in.i++;
		if (in.i+re->InitLength > in.mx) return 0;
	}
	ReStateNumber spaces[100];
	ReStateNumber *toFree;
	ReStateSet stateSet[2];
	if (re->States <= (sizeof(spaces)/(sizeof(spaces[0])*2)))
	{
		toFree = nullptr;
		stateSet[0].data = spaces;
	}
	else
	{
		toFree = (ReStateNumber *)_alloc(sizeof(ReStateNumber)*2*re->States);
		if (!toFree) return -1;
		stateSet[0].data = toFree;
	}
	stateSet[1].data = &stateSet[0].data[re->States];
	ReStateSet *next = &stateSet[1];
	next->length = 0;
	re_add_state(next, 0);
	int c = RE_EOF+1;
	int prev = 0;
	unsigned int swap = 0;
	unsigned int i;
	int rc = 0;
	ReStateSet *this_;
	while (c != RE_EOF && next->length > 0)
	{
		prev = c;
		c = re->NextChar(&in);
		this_ = next;
		next = &stateSet[swap];
		swap = 1 - swap;
		next->length = 0;
		for (i = 0; i < this_->length; i++)
		{
			int x = this_->data[i];
			switch (re->OPs[x])
			{
			case RE_OP_MATCH: {
				if (re->Args[x] == c) re_add_state(next, x+1);
				break; }
			case RE_OP_ANY: {
				re_add_state(next, x+1);
				break; }
			case RE_OP_WORD: {
				if (re_word_char(c)) re_add_state(next, x+1);
				break; }
			case RE_OP_NOTWORD: {
				if (!re_word_char(c)) re_add_state(next, x+1);
				break; }
			case RE_OP_DIGIT: {
				if (re_digit_char(c)) re_add_state(next, x+1);
				break; }
			case RE_OP_NOTDIGIT: {
				if (!re_digit_char(c)) re_add_state(next, x+1);
				break; }
			case RE_OP_SPACE: {
				if (re_space_char(c)) re_add_state(next, x+1);
				break; }
			case RE_OP_NOTSPACE: {
				if (!re_space_char(c)) re_add_state(next, x+1);
				break; }
			case RE_OP_BOUNDARY: {
				if (re_word_char(c) != re_word_char(prev)) re_add_state(this_, x+1);
				break; }
			case RE_OP_ANYSTAR: {
				re_add_state(next, x);
				re_add_state(this_, x+1);
				break; }
			case RE_OP_FORK: {
				re_add_state(this_, x+re->Args[x]);
				re_add_state(this_, x+1);
				break; }
			case RE_OP_GOTO: {
				re_add_state(this_, x+re->Args[x]);
				break; }
			case RE_OP_ACCEPT: {
				rc = 1;
				goto re_match_end; }
			case RE_OP_CC_INC:
			case RE_OP_CC_EXC: {
				int j = 1;
				int n = re->Args[x];
				int hit = 0;
				for (j = 1; j > 0 && j < n; j++)
				{
					if (re->OPs[x+j] == RE_OP_CC_VALUE)
					{
						if (re->Args[x+j] == c) { hit = 1; j = -1; }
					}
					else
					{
						if (re->Args[x+j] <= c && re->Args[x+j+1] >= c) { hit = 1; j = -1; }
						else j++;
					}
				}
				if (re->OPs[x] == RE_OP_CC_EXC) hit = !hit;
				if (hit) re_add_state(next, x+n);
				break; }
			}
		}
	}
	for (i = 0; i < next->length; i++)
	{
		if (re->OPs[next->data[i]] == RE_OP_ACCEPT) { rc = 1; break; }
	}
re_match_end:
	_free(toFree);
	return rc;
}

// Resize the opcode and argument arrays for an RE under construction.
__device__ static int re_resize(ReCompiled *p, int N)
{
	char *ops = (char *)_realloc(p->OPs, N*sizeof(p->OPs[0]));
	if (!ops) return 1;
	p->OPs = ops;
	int *args = (int *)_realloc(p->Args, N*sizeof(p->Args[0]));
	if (!args) return 1;
	p->Args = args;
	p->Allocs = N;
	return 0;
}

// Insert a new opcode and argument into an RE under construction.  The insertion point is just prior to existing opcode iBefore.
__device__ static int re_insert(ReCompiled *p, int before, int op, int arg)
{
	if (p->Allocs <= p->States && re_resize(p, p->Allocs*2)) return 0;
	for (int i = p->States; i > before; i--)
	{
		p->OPs[i] = p->OPs[i-1];
		p->Args[i] = p->Args[i-1];
	}
	p->States++;
	p->OPs[before] = op;
	p->Args[before] = arg;
	return before;
}

// Append a new opcode and argument to the end of the RE under construction.
__device__ static int re_append(ReCompiled *p, int op, int arg) { return re_insert(p, p->States, op, arg); }

// Make a copy of N opcodes starting at iStart onto the end of the RE under construction.
__device__ static void re_copy(ReCompiled *p, int start, int n)
{
	if (p->States+n >= p->Allocs && re_resize(p, p->Allocs*2+n)) return;
	_memcpy(&p->OPs[p->States], &p->OPs[start], n*sizeof(p->OPs[0]));
	_memcpy(&p->Args[p->States], &p->Args[start], n*sizeof(p->Args[0]));
	p->States += n;
}

// Return true if c is a hexadecimal digit character:  [0-9a-fA-F] If c is a hex digit, also set *pV = (*pV)*16 + valueof(c).  If
// c is not a hex digit *pV is unchanged.
__device__ static int re_hex(int c, int *v)
{
	if (c >= '0' && c <= '9') c -= '0';
	else if (c>='a' && c <= 'f') c -= 'a' - 10;
	else if (c>='A' && c <= 'F') c -= 'A' - 10;
	else return 0;
	*v = (*v)*16 + (c & 0xff);
	return 1;
}

// A backslash character has been seen, read the next character and return its interpretation.
__constant__ static const char _esc[] = "afnrtv\\()*.+?[$^{|}]";
__constant__ static const char _trans[] = "\a\f\n\r\t\v";
__device__ static unsigned re_esc_char(ReCompiled *p)
{
	if (p->In.i >= p->In.mx) return 0;
	char c = p->In.z[p->In.i];
	int v = 0;
	if (c == 'u' && p->In.i+4 < p->In.mx)
	{
		const unsigned char *in_ = p->In.z + p->In.i;
		if (re_hex(in_[1], &v) && re_hex(in_[2], &v) && re_hex(in_[3], &v) && re_hex(in_[4], &v))
		{
			p->In.i += 5;
			return v;
		}
	}
	if (c == 'x' && p->In.i+2 < p->In.mx)
	{
		const unsigned char *in_ = p->In.z + p->In.i;
		if (re_hex(in_[1], &v) && re_hex(in_[2], &v))
		{
			p->In.i += 3;
			return v;
		}
	}
	int i;
	for (i = 0; _esc[i] && _esc[i]!=c; i++) { }
	if (_esc[i])
	{
		if (i < 6) c = _trans[i];
		p->In.i++;
	}
	else
		p->Err = "unknown \\ escape";
	return c;
}

// Forward declaration
__device__ static const char *re_subcompile_string(ReCompiled *);

// Peek at the next byte of input
__device__ static unsigned char rePeek(ReCompiled *p) { return (p->In.i < p->In.mx ? p->In.z[p->In.i] : 0); }

// Compile RE text into a sequence of opcodes.  Continue up to the first unmatched ")" character, then return.  If an error is found,
// return a pointer to the error message string.
__device__ static const char *re_subcompile_re(ReCompiled *p)
{
	int start = p->States;
	const char *err = re_subcompile_string(p);
	if (err) return err;
	while (rePeek(p) == '|')
	{
		int end = p->States;
		re_insert(p, start, RE_OP_FORK, end + 2 - start);
		int goto_ = re_append(p, RE_OP_GOTO, 0);
		p->In.i++;
		err = re_subcompile_string(p);
		if (err) return err;
		p->Args[goto_] = p->States - goto_;
	}
	return 0;
}

// Compile an element of regular expression text (anything that can be an operand to the "|" operator).  Return NULL on success or a pointer
// to the error message if there is a problem.
__device__ static const char *re_subcompile_string(ReCompiled *p)
{
	int prev = -1;
	unsigned c;
	const char *err;
	while ((c = p->NextChar(&p->In)) != 0)
	{
		int start = p->States;
		switch (c)
		{
		case '|':
		case '$': 
		case ')': {
			p->In.i--;
			return 0; }
		case '(': {
			err = re_subcompile_re(p);
			if (err) return err;
			if (rePeek(p) != ')') return "unmatched '('";
			p->In.i++;
			break; }
		case '.': {
			if (rePeek(p) == '*')
			{
				re_append(p, RE_OP_ANYSTAR, 0);
				p->In.i++;
			}
			else
				re_append(p, RE_OP_ANY, 0);
			break; }
		case '*': {
			if (prev < 0) return "'*' without operand";
			re_insert(p, prev, RE_OP_GOTO, p->States - prev + 1);
			re_append(p, RE_OP_FORK, prev - p->States + 1);
			break; }
		case '+': {
			if (prev < 0) return "'+' without operand";
			re_append(p, RE_OP_FORK, prev - p->States);
			break; }
		case '?': {
			if (prev < 0) return "'?' without operand";
			re_insert(p, prev, RE_OP_FORK, p->States - prev+1);
			break; }
		case '{': {
			if (prev < 0) return "'{m,n}' without operand";
			int m = 0;
			while ((c = rePeek(p)) >= '0' && c <= '9') { m = m*10 + c - '0'; p->In.i++; }
			int n = m;
			if (c == ',')
			{
				p->In.i++;
				n = 0;
				while ((c = rePeek(p)) >= '0' && c <= '9') { n = n*10 + c-'0'; p->In.i++; }
			}
			if (c != '}') return "unmatched '{'";
			if (n > 0 && n < m) return "n less than m in '{m,n}'";
			p->In.i++;
			int j, sz = p->States - prev;
			if (m == 0)
			{
				if (n == 0) return "both m and n are zero in '{m,n}'";
				re_insert(p, prev, RE_OP_FORK, sz+1);
				n--;
			}
			else
			{
				for (j = 1; j < m; j++) re_copy(p, prev, sz);
			}
			for (j = m; j < n; j++)
			{
				re_append(p, RE_OP_FORK, sz+1);
				re_copy(p, prev, sz);
			}
			if (n == 0 && m > 0)
				re_append(p, RE_OP_FORK, -sz);
			break; }
		case '[': {
			int first = p->States;
			if (rePeek(p) == '^')
			{
				re_append(p, RE_OP_CC_EXC, 0);
				p->In.i++;
			}
			else
				re_append(p, RE_OP_CC_INC, 0);
			while ((c = p->NextChar(&p->In)) != 0)
			{
				if (c == '[' && rePeek(p) == ':')
					return "POSIX character classes not supported";
				if (c == '\\') c = re_esc_char(p);
				if (rePeek(p) == '-')
				{
					re_append(p, RE_OP_CC_RANGE, c);
					p->In.i++;
					c = p->NextChar(&p->In);
					if (c == '\\') c = re_esc_char(p);
					re_append(p, RE_OP_CC_RANGE, c);
				}
				else
					re_append(p, RE_OP_CC_VALUE, c);
				if (rePeek(p) == ']') { p->In.i++; break; }
			}
			if (c == 0) return "unclosed '['";
			p->Args[first] = p->States - first;
			break; }
		case '\\': {
			int specialOp = 0;
			switch (rePeek(p))
			{
			case 'b': specialOp = RE_OP_BOUNDARY;   break;
			case 'd': specialOp = RE_OP_DIGIT;      break;
			case 'D': specialOp = RE_OP_NOTDIGIT;   break;
			case 's': specialOp = RE_OP_SPACE;      break;
			case 'S': specialOp = RE_OP_NOTSPACE;   break;
			case 'w': specialOp = RE_OP_WORD;       break;
			case 'W': specialOp = RE_OP_NOTWORD;    break;
			}
			if (specialOp)
			{
				p->In.i++;
				re_append(p, specialOp, 0);
			}
			else
			{
				c = re_esc_char(p);
				re_append(p, RE_OP_MATCH, c);
			}
			break; }
		default: {
			re_append(p, RE_OP_MATCH, c);
			break; }
		}
		prev = start;
	}
	return 0;
}

// Free and reclaim all the memory used by a previously compiled regular expression.  Applications should invoke this routine once
// for every call to re_compile() to avoid memory leaks.
void re_free(ReCompiled *re)
{
	if (re)
	{
		_free(re->OPs);
		_free(re->Args);
		_free(re);
	}
}

// Compile a textual regular expression in zIn[] into a compiled regular expression suitable for us by re_match() and return a pointer to the
// compiled regular expression in *pre.  Return NULL on success or an error message if something goes wrong.
__device__ const char *re_compile(ReCompiled **pre, const char *in_, int noCase)
{
	*pre = nullptr;
	ReCompiled *re = (ReCompiled *)_alloc(sizeof(*re));
	if (!re)
		return "out of memory";
	_memset(re, 0, sizeof(*re));
	re->NextChar = (noCase ? re_next_char_nocase : re_next_char);
	if (re_resize(re, 30))
	{
		re_free(re);
		return "out of memory";
	}
	if (in_[0] == '^')
		in_++;
	else
		re_append(re, RE_OP_ANYSTAR, 0);
	re->In.z = (unsigned char*)in_;
	re->In.i = 0;
	re->In.mx = (int)_strlen(in_);
	const char *err = re_subcompile_re(re);
	if (err)
	{
		re_free(re);
		return err;
	}
	if (rePeek(re) == '$' && re->In.i+1 >= re->In.mx)
	{
		re_append(re, RE_OP_MATCH, RE_EOF);
		re_append(re, RE_OP_ACCEPT, 0);
		*pre = re;
	}
	else if (re->In.i >= re->In.mx)
	{
		re_append(re, RE_OP_ACCEPT, 0);
		*pre = re;
	}
	else
	{
		re_free(re);
		return "unrecognized character";
	}

	// The following is a performance optimization.  If the regex begins with ".*" (if the input regex lacks an initial "^") and afterwards there are
	// one or more matching characters, enter those matching characters into zInit[].  The re_match() routine can then search ahead in the input 
	// string looking for the initial match without having to run the whole regex engine over the string.  Do not worry able trying to match
	// unicode characters beyond plane 0 - those are very rare and this is just an optimization.
	if (re->OPs[0] == RE_OP_ANYSTAR)
	{
		int i, j;
		for (j = 0, i = 1; j < sizeof(re->Init)-2 && re->OPs[i] == RE_OP_MATCH; i++)
		{
			unsigned x = re->Args[i];
			if (x <= 127)
				re->Init[j++] = x;
			else if (x <= 0xfff)
			{
				re->Init[j++] = 0xc0 | (x>>6);
				re->Init[j++] = 0x80 | (x&0x3f);
			}
			else if (x <= 0xffff)
			{
				re->Init[j++] = 0xd0 | (x>>12);
				re->Init[j++] = 0x80 | ((x>>6)&0x3f);
				re->Init[j++] = 0x80 | (x&0x3f);
			}
			else
				break;
		}
		if (j > 0 && re->Init[j-1] == 0) j--;
		re->InitLength = j;
	}
	return re->Err;
}

// Implementation of the regexp() SQL function.  This function implements the build-in REGEXP operator.  The first argument to the function is the
// pattern and the second argument is the string.  So, the SQL statements:
//       A REGEXP B
//
// is implemented as regexp(B,A).
__device__ static void re_sql_func(FuncContext *fctx, int argc, Mem **args)
{
	ReCompiled *re = (ReCompiled *)Vdbe::get_Auxdata(fctx, 0); // Compiled regular expression
	if (!re)
	{
		const char *pattern = (const char *)Vdbe::Value_Text(args[0]); // The regular expression
		if (!pattern) return;
		const char *err = re_compile(&re, pattern, 0); // Compile error message
		if (err)
		{
			re_free(re);
			Vdbe::Result_Error(fctx, err, -1);
			return;
		}
		if (!re)
		{
			Vdbe::Result_ErrorNoMem(fctx);
			return;
		}
		Vdbe::set_Auxdata(fctx, 0, re, (void(*)(void*))re_free);
	}
	const unsigned char *str = (const unsigned char *)Vdbe::Value_Text(args[1]); // String being searched
	if (str)
		Vdbe::Result_Int(fctx, re_match(re, str, -1));
}

// Invoke this routine in order to install the REGEXP function in an SQLite database connection.
// Use:
//      sqlite3_auto_extension(sqlite3_add_regexp_func);
//
// to cause this extension to be automatically loaded into each new database connection.
__device__ int sqlite3_add_regexp_func(Context *ctx)
{
	return Main::CreateFunction(ctx, "regexp", 2, TEXTENCODE_UTF8, nullptr, re_sql_func, nullptr, nullptr);
}

#pragma region Test Code
#ifdef _TEST
#include <JimEx.h>
__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);

// Implementation of the TCL command:
//      sqlite3_add_regexp_func $DB
__device__ static int tclSqlite3AddRegexpFunc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	sqlite3_add_regexp_func(ctx);
	return JIM_OK;
}

// Register the sqlite3_add_regexp_func TCL command with the TCL interpreter.
__device__ int Sqlitetestregexp_Init(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "sqlite3_add_regexp_func", tclSqlite3AddRegexpFunc, nullptr, nullptr);
	return JIM_OK;
}

#endif
#pragma endregion

