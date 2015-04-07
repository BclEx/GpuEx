#include <stdio.h>
#include <stdlib.h>


// identifies the kind of regular expression
#define ALT 0 
#define CON 1
#define KLN 2
#define LIT 3
#define EPS 4

#define TOTAL_KINDS 5 // total number of kinds

/* regular expression structure used by the parser on the host side */
typedef struct regex
{
	/* kind of expression */
	int kind;

	/* fields populated based on the kind of expression */
	struct regex* e_one;
	struct regex* e_two;
	char c;

	/* continuation pointer */
	struct regex* k;

	/* index assigned to this node */
	int idx;
} regex;

/* regular expression structure optimized for the device side */
typedef struct dregex
{
	/* kind of expression */
	int kind;

	/* fields populated based on the kind of expression */
	int e_one;
	int e_two;
	char c;

	/* continuation */
	int k;
} dregex;

/* utility method for initializing a new regex */
regex* new_regex();

/* builds a regular expression parse tree */
bool parse_regex(char* input, regex** result, int** kind_stats);

/* utility method for deallocating a host expression */
void free_hexp(regex* exp);
