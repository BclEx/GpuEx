#if 0
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <setjmp.h>
#include "Runtime.h"
#include "Regex.h"

#ifdef _DEBUG
#include <stdio.h>

__device__ static const char_t *g_nnames[] =
{
	_L("NONE"),_L("OP_GREEDY"),	_L("OP_OR"),
	_L("OP_EXPR"),_L("OP_NOCAPEXPR"),_L("OP_DOT"),	_L("OP_CLASS"),
	_L("OP_CCLASS"),_L("OP_NCLASS"),_L("OP_RANGE"),_L("OP_CHAR"),
	_L("OP_EOL"),_L("OP_BOL"),_L("OP_WB")
};

#endif
#define OP_GREEDY		(MAX_CHAR+1) // * + ? {n}
#define OP_OR			(MAX_CHAR+2)
#define OP_EXPR			(MAX_CHAR+3) //parentesis ()
#define OP_NOCAPEXPR	(MAX_CHAR+4) //parentesis (?:)
#define OP_DOT			(MAX_CHAR+5)
#define OP_CLASS		(MAX_CHAR+6)
#define OP_CCLASS		(MAX_CHAR+7)
#define OP_NCLASS		(MAX_CHAR+8) //negates class the [^
#define OP_RANGE		(MAX_CHAR+9)
#define OP_CHAR			(MAX_CHAR+10)
#define OP_EOL			(MAX_CHAR+11)
#define OP_BOL			(MAX_CHAR+12)
#define OP_WB			(MAX_CHAR+13)

#define REG_SYMBOL_ANY_CHAR ('.')
#define REG_SYMBOL_GREEDY_ONE_OR_MORE ('+')
#define REG_SYMBOL_GREEDY_ZERO_OR_MORE ('*')
#define REG_SYMBOL_GREEDY_ZERO_OR_ONE ('?')
#define REG_SYMBOL_BRANCH ('|')
#define REG_SYMBOL_END_OF_STRING ('$')
#define REG_SYMBOL_BEGINNING_OF_STRING ('^')
#define REG_SYMBOL_ESCAPE_CHAR ('\\')

typedef int NodeType;

typedef struct tagNode
{
	NodeType type;
	int left;
	int right;
	int next;
} node_t;

struct regex_t
{
	const char_t *_eol;
	const char_t *_bol;
	const char_t *_p;
	int _first;
	int _op;
	node_t *_nodes;
	int _nallocated;
	int _nsize;
	int _nsubexpr;
	regmatch_t *_matches;
	int _currsubexp;
	void *_jmpbuf;
	const char_t **_error;
};

__device__ static int reg_list(regex_t *exp);

__device__ static int reg_newnode(regex_t *exp, NodeType type)
{
	node_t n;
	int newid;
	n.type = type;
	n.next = n.right = n.left = -1;
	if (type == OP_EXPR)
		n.right = exp->_nsubexpr++;
	if (exp->_nallocated < (exp->_nsize + 1))
	{
		//int oldsize = exp->_nallocated;
		exp->_nallocated *= 2;
		exp->_nodes = (node_t *)_realloc(exp->_nodes, exp->_nallocated * sizeof(node_t));
	}
	exp->_nodes[exp->_nsize++] = n;
	newid = exp->_nsize - 1;
	return (int)newid;
}

__device__ static void reg_error(regex_t *exp, const char_t *error)
{
	if (exp->_error) *exp->_error = error;
	longjmp(*((jmp_buf*)exp->_jmpbuf), -1);
}

__device__ static void reg_expect(regex_t *exp, int n)
{
	if ((*exp->_p) != n)
		reg_error(exp, _L("expected paren"));
	exp->_p++;
}

__device__ static char_t reg_escapechar(regex_t *exp)
{
	if (*exp->_p == REG_SYMBOL_ESCAPE_CHAR)
	{
		exp->_p++;
		switch (*exp->_p)
		{
		case 'v': exp->_p++; return '\v';
		case 'n': exp->_p++; return '\n';
		case 't': exp->_p++; return '\t';
		case 'r': exp->_p++; return '\r';
		case 'f': exp->_p++; return '\f';
		default: return (*exp->_p++);
		}
	} else if (!_isprint(*exp->_p)) reg_error(exp,_L("letter expected"));
	return (*exp->_p++);
}

__device__ static int reg_charclass(regex_t *exp, int classid)
{
	int n = reg_newnode(exp,OP_CCLASS);
	exp->_nodes[n].left = classid;
	return n;
}

__device__ static int reg_charnode(regex_t *exp,bool isclass)
{
	char_t t;
	if (*exp->_p == REG_SYMBOL_ESCAPE_CHAR)
	{
		exp->_p++;
		switch (*exp->_p)
		{
		case 'n': exp->_p++; return reg_newnode(exp, '\n');
		case 't': exp->_p++; return reg_newnode(exp, '\t');
		case 'r': exp->_p++; return reg_newnode(exp, '\r');
		case 'f': exp->_p++; return reg_newnode(exp, '\f');
		case 'v': exp->_p++; return reg_newnode(exp, '\v');
		case 'a': case 'A': case 'w': case 'W': case 's': case 'S':
		case 'd': case 'D': case 'x': case 'X': case 'c': case 'C':
		case 'p': case 'P': case 'l': case 'u':
			{
				t = *exp->_p; exp->_p++;
				return reg_charclass(exp, t);
			}
		case 'b':
		case 'B':
			if (!isclass)
			{
				int node = reg_newnode(exp, OP_WB);
				exp->_nodes[node].left = *exp->_p;
				exp->_p++;
				return node;
			}
		default:
			t = *exp->_p; exp->_p++;
			return reg_newnode(exp, t);
		}
	}
	else if (!_isprint(*exp->_p))
		reg_error(exp, _L("letter expected"));
	t = *exp->_p; exp->_p++;
	return reg_newnode(exp, t);
}

__device__ static int reg_class(regex_t *exp)
{
	int ret = -1;
	int first = -1, chain;
	if (*exp->_p == REG_SYMBOL_BEGINNING_OF_STRING)
	{
		ret = reg_newnode(exp, OP_NCLASS);
		exp->_p++;
	}
	else ret = reg_newnode(exp, OP_CLASS);

	if (*exp->_p == ']') reg_error(exp, _L("empty class"));
	chain = ret;
	while (*exp->_p != ']' && exp->_p != exp->_eol)
	{
		if (*exp->_p == '-' && first != -1)
		{
			int r, t;
			if (*exp->_p++ == ']') reg_error(exp, _L("unfinished range"));
			r = reg_newnode(exp,OP_RANGE);
			if (first>*exp->_p) reg_error(exp, _L("invalid range"));
			if (exp->_nodes[first].type == OP_CCLASS) reg_error(exp, _L("cannot use character classes in ranges"));
			exp->_nodes[r].left = exp->_nodes[first].type;
			t = reg_escapechar(exp);
			exp->_nodes[r].right = t;
			exp->_nodes[chain].next = r;
			chain = r;
			first = -1;
		}
		else
		{
			if (first != -1)
			{
				int c = first;
				exp->_nodes[chain].next = c;
				chain = c;
				first = reg_charnode(exp,true);
			}
			else
				first = reg_charnode(exp,true);
		}
	}
	if (first != -1)
	{
		int c = first;
		exp->_nodes[chain].next = c;
		chain = c;
		first = -1;
	}
	// hack?
	exp->_nodes[ret].left = exp->_nodes[ret].next;
	exp->_nodes[ret].next = -1;
	return ret;
}

__device__ static int reg_parsenumber(regex_t *exp)
{
	int ret = *exp->_p-'0';
	int positions = 10;
	exp->_p++;
	while (_isdigit(*exp->_p))
	{
		ret = ret*10+(*exp->_p++-'0');
		if (positions==1000000000) reg_error(exp, _L("overflow in numeric constant"));
		positions *= 10;
	};
	return ret;
}

__device__ static int reg_element(regex_t *exp)
{
	int ret = -1;
	switch (*exp->_p)
	{
	case '(': {
		int expr, newn;
		exp->_p++;

		if (*exp->_p =='?')
		{
			exp->_p++;
			reg_expect(exp, ':');
			expr = reg_newnode(exp, OP_NOCAPEXPR);
		}
		else
			expr = reg_newnode(exp, OP_EXPR);
		newn = reg_list(exp);
		exp->_nodes[expr].left = newn;
		ret = expr;
		reg_expect(exp,')');
		break; }

	case '[':
		exp->_p++;
		ret = reg_class(exp);
		reg_expect(exp, ']');
		break;
	case REG_SYMBOL_END_OF_STRING: exp->_p++; ret = reg_newnode(exp,OP_EOL); break;
	case REG_SYMBOL_ANY_CHAR: exp->_p++; ret = reg_newnode(exp,OP_DOT); break;
	default:
		ret = reg_charnode(exp,false);
		break;
	}

	{
		bool isgreedy = false;
		unsigned short p0 = 0, p1 = 0;
		switch (*exp->_p)
		{
		case REG_SYMBOL_GREEDY_ZERO_OR_MORE: p0 = 0; p1 = 0xFFFF; exp->_p++; isgreedy = true; break;
		case REG_SYMBOL_GREEDY_ONE_OR_MORE: p0 = 1; p1 = 0xFFFF; exp->_p++; isgreedy = true; break;
		case REG_SYMBOL_GREEDY_ZERO_OR_ONE: p0 = 0; p1 = 1; exp->_p++; isgreedy = true; break;
		case '{':
			exp->_p++;
			if (!_isdigit(*exp->_p)) reg_error(exp, _L("number expected"));
			p0 = (unsigned short)reg_parsenumber(exp);
			/////////////////////////////////
			switch (*exp->_p)
			{
			case '}':
				p1 = p0; exp->_p++;
				break;
			case ',':
				exp->_p++;
				p1 = 0xFFFF;
				if (_isdigit(*exp->_p))
					p1 = (unsigned short)reg_parsenumber(exp);
				reg_expect(exp, '}');
				break;
			default:
				reg_error(exp, _L(", or } expected"));
			}
			/////////////////////////////////
			isgreedy = true;
			break;
		}
		if (isgreedy)
		{
			int nnode = reg_newnode(exp,OP_GREEDY);
			//int op = OP_GREEDY;
			exp->_nodes[nnode].left = ret;
			exp->_nodes[nnode].right = ((p0)<<16)|p1;
			ret = nnode;
		}
	}
	if ((*exp->_p != REG_SYMBOL_BRANCH) && (*exp->_p != ')') && (*exp->_p != REG_SYMBOL_GREEDY_ZERO_OR_MORE) && (*exp->_p != REG_SYMBOL_GREEDY_ONE_OR_MORE) && (*exp->_p != '\0'))
	{
		int nnode = reg_element(exp);
		exp->_nodes[ret].next = nnode;
	}
	return ret;
}

__device__ static int reg_list(regex_t *exp)
{
	int ret = -1, e;
	if (*exp->_p == REG_SYMBOL_BEGINNING_OF_STRING)
	{
		exp->_p++;
		ret = reg_newnode(exp, OP_BOL);
	}
	e = reg_element(exp);
	if (ret != -1)
		exp->_nodes[ret].next = e;
	else ret = e;

	if (*exp->_p == REG_SYMBOL_BRANCH)
	{
		int temp, tright;
		exp->_p++;
		temp = reg_newnode(exp, OP_OR);
		exp->_nodes[temp].left = ret;
		tright = reg_list(exp);
		exp->_nodes[temp].right = tright;
		ret = temp;
	}
	return ret;
}

__device__ static bool reg_matchcclass(int cclass, char_t c)
{
	switch (cclass)
	{
	case 'a': return _isalpha(c)?true:false;
	case 'A': return !_isalpha(c)?true:false;
	case 'w': return (_isalnum(c) || c == '_')?true:false;
	case 'W': return (!_isalnum(c) && c != '_')?true:false;
	case 's': return _isspace(c)?true:false;
	case 'S': return !_isspace(c)?true:false;
	case 'd': return _isdigit(c)?true:false;
	case 'D': return !_isdigit(c)?true:false;
	case 'x': return _isxdigit(c)?true:false;
	case 'X': return !_isxdigit(c)?true:false;
		//case 'c': return _iscntrl(c)?true:false;
		//case 'C': return !_iscntrl(c)?true:false;
		//case 'p': return _ispunct(c)?true:false;
		//case 'P': return !_ispunct(c)?true:false;
		//case 'l': return _islower(c)?true:false;
		//case 'u': return _isupper(c)?true:false;
	}
	return false; // cannot happen
}

__device__ static bool reg_matchclass(regex_t* exp, node_t *node, char_t c)
{
	do
	{
		switch (node->type)
		{
		case OP_RANGE:
			if (c >= node->left && c <= node->right) return true;
			break;
		case OP_CCLASS:
			if (reg_matchcclass(node->left, c)) return true;
			break;
		default:
			if (c == node->type) return true;
		}
	} while ((node->next != -1) && (node = &exp->_nodes[node->next]));
	return false;
}

__device__ static const char_t *reg_matchnode(regex_t* exp,node_t *node,const char_t *str,node_t *next)
{
	NodeType type = node->type;
	switch (type)
	{
	case OP_GREEDY: {
		//node_t *greedystop = (node->next != -1) ? &exp->_nodes[node->next] : nullptr;
		node_t *greedystop = nullptr;
		int p0 = (node->right >> 16)&0x0000FFFF, p1 = node->right&0x0000FFFF, nmaches = 0;
		const char_t *s=str, *good = str;

		greedystop = (node->next != -1 ? &exp->_nodes[node->next] : next);
		while ((nmaches == 0xFFFF || nmaches < p1))
		{
			const char_t *stop;
			if (!(s = reg_matchnode(exp, &exp->_nodes[node->left], s, greedystop)))
				break;
			nmaches++;
			good = s;
			if (greedystop)
			{
				// checks that 0 matches satisfy the expression(if so skips). if not would always stop(for instance if is a '?').
				if (greedystop->type != OP_GREEDY || (greedystop->type == OP_GREEDY && ((greedystop->right >> 16)&0x0000FFFF) != 0))
				{
					node_t *gnext = nullptr;
					if (greedystop->next != -1)
						gnext = &exp->_nodes[greedystop->next];
					else if (next && next->next != -1)
						gnext = &exp->_nodes[next->next];
					stop = reg_matchnode(exp,greedystop,s,gnext);
					if (stop)
					{
						// if satisfied stop it
						if (p0 == p1 && p0 == nmaches) break;
						else if (nmaches >= p0 && p1 == 0xFFFF) break;
						else if (nmaches >= p0 && nmaches <= p1) break;
					}
				}
			}

			if (s >= exp->_eol)
				break;
		}
		if (p0 == p1 && p0 == nmaches) return good;
		else if (nmaches >= p0 && p1 == 0xFFFF) return good;
		else if (nmaches >= p0 && nmaches <= p1) return good;
		return nullptr; }
	case OP_OR: {
		const char_t *asd = str;
		node_t *temp = &exp->_nodes[node->left];
		while ((asd = reg_matchnode(exp, temp, asd, nullptr)))
			if (temp->next != -1)
				temp = &exp->_nodes[temp->next];
			else
				return asd;
		asd = str;
		temp = &exp->_nodes[node->right];
		while ((asd = reg_matchnode(exp, temp, asd, nullptr)))
			if (temp->next != -1)
				temp = &exp->_nodes[temp->next];
			else
				return asd;
		return nullptr; }
	case OP_EXPR:
	case OP_NOCAPEXPR: {
		node_t *n = &exp->_nodes[node->left];
		const char_t *cur = str;
		int capture = -1;
		if (node->type != OP_NOCAPEXPR && node->right == exp->_currsubexp)
		{
			capture = exp->_currsubexp;
			exp->_matches[capture].begin = cur;
			exp->_currsubexp++;
		}

		do
		{
			node_t *subnext = nullptr;
			subnext = (n->next != -1 ? &exp->_nodes[n->next] : next);
			if (!(cur = reg_matchnode(exp,n,cur,subnext)))
			{
				if (capture != -1)
				{
					exp->_matches[capture].begin = 0;
					exp->_matches[capture].len = 0;
				}
				return nullptr;
			}
		} while ((n->next != -1) && (n = &exp->_nodes[n->next]));

		if (capture != -1)
			exp->_matches[capture].len = cur - exp->_matches[capture].begin;
		return cur; }
	case OP_WB:
		if (str == exp->_bol && !_isspace(*str)
			|| (str == exp->_eol && !_isspace(*(str-1)))
			|| (!_isspace(*str) && _isspace(*(str+1)))
			|| (_isspace(*str) && !_isspace(*(str+1))))
			return (node->left == 'b' ? str : nullptr);
		return (node->left == 'b' ? nullptr : str);
	case OP_BOL:
		if (str == exp->_bol) return str;
		return nullptr;
	case OP_EOL:
		if (str == exp->_eol) return str;
		return nullptr;
	case OP_DOT: {
		*str++;
		return str; }
	case OP_NCLASS:
	case OP_CLASS:
		if (reg_matchclass(exp, &exp->_nodes[node->left], *str) ? (type == OP_CLASS) : (type == OP_NCLASS))
		{
			*str++;
			return str;
		}
		return nullptr;
	case OP_CCLASS:
		if (reg_matchcclass(node->left,*str))
		{
			*str++;
			return str;
		}
		return nullptr;
	default: // char
		if (*str != node->type) return nullptr;
		*str++;
		return str;
	}
	return nullptr;
}

// public api
__device__ regex_t *reg_compile(const char_t *pattern, const char_t **error)
{
	regex_t *exp = (regex_t *)_alloc(sizeof(regex_t));
	exp->_eol = exp->_bol = nullptr;
	exp->_p = pattern;
	exp->_nallocated = (int)_strlen(pattern) * sizeof(char_t);
	exp->_nodes = (node_t *)_alloc(exp->_nallocated * sizeof(node_t));
	exp->_nsize = 0;
	exp->_matches = 0;
	exp->_nsubexpr = 0;
	exp->_first = reg_newnode(exp,OP_EXPR);
	exp->_error = error;
	exp->_jmpbuf = _alloc(sizeof(jmp_buf));
	if (setjmp(*((jmp_buf*)exp->_jmpbuf)) == 0) {
		int res = reg_list(exp);
		exp->_nodes[exp->_first].left = res;
		if (*exp->_p!='\0')
			reg_error(exp,_L("unexpected character"));
#ifdef _DEBUG
		{
			int nsize,i;
			node_t *t;
			nsize = exp->_nsize;
			t = &exp->_nodes[0];
			_printf(_L("\n"));
			for (i = 0;i < nsize; i++)
			{
				if (exp->_nodes[i].type>MAX_CHAR)
					_printf(_L("[%02d] %10s "), i, g_nnames[exp->_nodes[i].type-MAX_CHAR]);
				else
					_printf(_L("[%02d] %10c "), i, exp->_nodes[i].type);
				_printf(_L("left %02d right %02d next %02d\n"), exp->_nodes[i].left,exp->_nodes[i].right, exp->_nodes[i].next);
			}
			_printf(_L("\n"));
		}
#endif
		exp->_matches = (regmatch_t *)_alloc(exp->_nsubexpr * sizeof(regmatch_t));
		_memset(exp->_matches, 0, exp->_nsubexpr * sizeof(regmatch_t));
	}
	else
	{
		reg_free(exp);
		return nullptr;
	}
	return exp;
}

__device__ void reg_free(regex_t *exp)
{
	if (exp)
	{
		if (exp->_nodes) _free(exp->_nodes);
		if (exp->_jmpbuf) _free(exp->_jmpbuf);
		if (exp->_matches) _free(exp->_matches);
		_free(exp);
	}
}

__device__ bool reg_match(regex_t* exp,const char_t* text)
{
	const char_t* res = nullptr;
	exp->_bol = text;
	exp->_eol = text + _strlen(text);
	exp->_currsubexp = 0;
	res = reg_matchnode(exp, exp->_nodes, text, nullptr);

#ifdef _DEBUG
	_printf("DEBUG reg_match: res = '%s'\n", res);
	_printf("DEBUG reg_match: exp->_eol = '%s'\n", exp->_eol);
#endif

	// Fail match if reg_matchnode returns nothing
	if (!res)
		return false;
	return true;
}

__device__ bool reg_searchrange(regex_t* exp, const char_t* text_begin, const char_t* text_end, const char_t** out_begin, const char_t** out_end)
{
	const char_t *cur = nullptr;
	int node = exp->_first;
	if (text_begin >= text_end) return false;
	exp->_bol = text_begin;
	exp->_eol = text_end;
	do
	{
		cur = text_begin;
		while (node != -1)
		{
			exp->_currsubexp = 0;
			cur = reg_matchnode(exp, &exp->_nodes[node], cur, nullptr);
			if (!cur)
				break;
			node = exp->_nodes[node].next;
		}
		*text_begin++;
	} while (cur == nullptr && text_begin != text_end);

	if (cur == nullptr)
		return false;

	--text_begin;

	if (out_begin) *out_begin = text_begin;
	if (out_end) *out_end = cur;
	return true;
}

__device__ bool reg_search(regex_t* exp,const char_t* text, const char_t** out_begin, const char_t** out_end)
{
	return reg_searchrange(exp, text, text + _strlen(text), out_begin, out_end);
}

__device__ int reg_getsubexpcount(regex_t* exp)
{
	return exp->_nsubexpr;
}

__device__ bool reg_getsubexp(regex_t* exp, int n, regmatch_t *subexp)
{
	if (n < 0 || n >= exp->_nsubexpr) return false;
	*subexp = exp->_matches[n];
	return true;
}

//// Regex must have exactly one bracket pair
//__device__ static char *reg_replace(const char *regex, const char *buf, const char *sub)
//{
//	char *s = nullptr;
//	int n, n1, n2, n3, s_len, len = _strlen(buf);
//	struct regmatch_t cap = { nullptr, 0 };
//	do
//	{
//		s_len = (s == nullptr ? 0 : _strlen(s));
//		if ((n = reg_match(regex, buf, len, &cap, 1, 0)) > 0)
//			n1 = cap.ptr - buf, n2 = _strlen(sub), n3 = &buf[n] - &cap.ptr[cap.len];
//		else
//			n1 = len, n2 = 0, n3 = 0;
//		s = (char *)_realloc(s, s_len + n1 + n2 + n3 + 1);
//		_memcpy(s + s_len, buf, n1);
//		_memcpy(s + s_len + n1, sub, n2);
//		_memcpy(s + s_len + n1 + n2, cap.ptr + cap.len, n3);
//		s[s_len + n1 + n2 + n3] = '\0';
//
//		buf += (n > 0 ? n : len);
//		len -= (n > 0 ? n : len);
//	} while (len > 0);
//	return s;
//}
#endif