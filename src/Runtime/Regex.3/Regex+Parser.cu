#include "regex.h"

/* utility method for initializing a new regex */
regex* new_regex() {
	regex* r = (regex*) malloc(sizeof(regex));
	r->kind = ALT;
	r->e_one = NULL;
	r->e_two = NULL;
	r->k = NULL;
	r->idx = -1;
	r->c = '\0';
	return r;
}

/* utility method for deallocating a host expression */
void free_hexp(regex* exp) {
	switch (exp->kind) {
		case ALT:
		case CON: {
			free_hexp(exp->e_one);
			free_hexp(exp->e_two);
			break;
		}
		case KLN: {
			free_hexp(exp->e_one);
			break;
		}
		default: {
			// Nothing needs to be done.
		}
	}
	free(exp);
}


/* token types */
typedef enum token {
	bar,
	asterix,
	literal,
	empstr, // @
	lbrace,
	rbrace,
	eof
} token;

/* token value (if applicable) */
int _tval;
char* _input;
int* _kind_stats;

/* returns the next token in stream */
token next_token() {
	// once we hit newline there's is more
	if (_tval == '\0') {
		return eof;
	}

	_tval = *_input++;

	switch (_tval) {
		case '|': 
			return bar;
		case '*': 
			return asterix;
		case '@': 
			return empstr;
		case '(':
			return lbrace;
		case ')':
			return rbrace;
		case '\0':
			return eof;
		case '\\':
			_tval = *_input++;
			return (_tval == '\0') ? eof : literal;
		default:
			return literal;
	}
}

/* returns the next token without consuming the input stream */
token look_ahead() {
	token result = next_token();

	// push back the character
	_input--;

	// handle character escapes carefully
	if (result == literal) {
		switch (_tval) {
			case '|':
			case '*':
			case '(':
			case ')':
				_input--;
				break;
		}
	}

	return result;
}

/* consumes the expected token */
bool eat(token expected, char* error_msg) {
	token tk = next_token();
	if (tk != expected) {
		printf("%s\n", error_msg);
		return false;
	}
	return true;
}

/* forward declaration */
regex* parse_exp();

/* an atom */
regex* parse_atom() {
	token tk = next_token();
	switch (tk) {
		case literal: {
			regex* result = new_regex();
			result->kind = LIT;
			result->c = (char) _tval;
			_kind_stats[LIT]++; 
			return result;
		}
		case empstr: {
			regex* result = new_regex();
			result->kind = EPS;
			_kind_stats[EPS]++; 
			return result;
		}
		case lbrace: {
			regex* result = parse_exp();
			if (NULL != result) {
				if (eat(rbrace, "PARSE ERROR: Missing closing parentheses.")) {
					return result;
				} else {
					free_hexp(result);
				}
			}
			break;
		}
		case eof: {
			printf("PARSE ERROR: Unexpected end of input.\n");
			break;
		}
		default: {
			printf("PARSE ERROR: Unexpected input character: [%c]\n", _tval);
		}	
	}
	
	return NULL;
}

/* an atom or a kleene of an atom */
regex* parse_factor() {
	regex* atom = parse_atom();
	if (NULL != atom) {
		token tk = look_ahead();
		if (tk == asterix) {
			eat(tk, NULL);
			regex* result = new_regex();
			result->kind = KLN;
			result->e_one = atom;
			_kind_stats[KLN]++; 
			return result;	
		}
		return atom;
	}
	return NULL;
}

/* series of concatenations of factors */
regex* parse_term() {
	regex* factor = parse_factor();

	if (NULL != factor) {
		token tk = look_ahead();
		switch (tk) {
			case eof:
			case bar:
			case rbrace:
				return factor;
			default: {
				regex* next_term = parse_term();
				if (NULL != next_term) {
					regex* result = new_regex();
					result->kind = CON;
					result->e_one = factor;
					result->e_two = next_term;
					_kind_stats[CON]++; 
					return result;
				} else {
					free_hexp(factor);
				}
			}
		}
	}

	return NULL;
}

/* series of alternations of terms */
regex* parse_exp() {
	regex* term = parse_term();

	if (NULL != term) {
		token tk = look_ahead();
		if (tk == bar) {
			eat(tk, NULL);
			regex* next_regex = parse_exp();
			if (NULL != next_regex) {
				regex* result = new_regex();
				result->kind = ALT;
				result->e_one = term;
				result->e_two = next_regex;
				_kind_stats[ALT]++; 
				return result;
			} else {
				free_hexp(term);
			}
		} else {
			return term;
		}
	}

	return NULL;
}

/* builds a regular expression parse tree from the given input */
bool parse_regex(char* input, regex** result, int** kind_stats) {
	_input = input;
	_tval = -1;
	_kind_stats = (int*) calloc(TOTAL_KINDS, sizeof(int));
	regex* _result = parse_exp();
	if (NULL != _result) {
		*result = _result;
		*kind_stats = _kind_stats;
		return true;
	}
	free(_kind_stats);
	return false;
}
