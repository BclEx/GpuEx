#include <Runtime.h>
#include "futils.h"
#include <ctype.h>

// See if the specified word is found in the specified string.
static bool search(char *string, char *word, bool ignorecase)
{
	int len = strlen(word);
	if (!ignorecase)
	{
		while (true)
		{
			string = strchr(string, word[0]);
			if (!string)
				return false;
			if (!memcmp(string, word, len))
				return true;
			string++;
		}
	}
	// Here if we need to check case independence. Do the search by lower casing both strings.
	int lowfirst = *word;
	if (isupper(lowfirst))
		lowfirst = tolower(lowfirst);
	while (true)
	{
		while (*string && *string != lowfirst && (!isupper(*string) || tolower(*string) != lowfirst))
			string++;
		if (*string == '\0')
			return false;
		char *cp1 = string;
		char *cp2 = word;
		int	ch1, ch2;
		do
		{
			if (*cp2 == '\0')
				return true;
			ch1 = *cp1++;
			if (isupper(ch1))
				ch1 = tolower(ch1);
			ch2 = *cp2++;
			if (isupper(ch2))
				ch2 = tolower(ch2);
		} while (ch1 == ch2);
		string++;
	}
}

int main(int argc, char **argv)
{
	argc--;
	argv++;
	bool ignorecase = false;
	bool tellline = false;
	if (**argv == '-')
	{
		argc--;
		char *cp = *argv++;
		while (*++cp) switch (*cp)
		{
		case 'i': ignorecase = TRUE; break;
		case 'n': tellline = TRUE; break;
		default: fprintf(stderr, "Unknown option\n"); exit(1);
		}
	}
	//
	char *word = *argv++;
	argc--;
	bool tellname = (argc > 1);
	//
	while (argc-- > 0)
	{
		char *name = *argv++;
		FILE *f = _fopen(name, "r");
		if (!f)
		{
			perror(name);
			continue;
		}
		long line = 0;
		char buf[8192];
		while (_fgets(buf, sizeof(buf), f))
		{
			char *cp = &buf[strlen(buf) - 1];
			if (*cp != '\n')
				fprintf(stderr, "%s: Line too long\n", name);
			if (search(buf, word, ignorecase))
			{
				if (tellname) printf("%s: ", name);
				if (tellline) printf("%d: ", line);
				_fputs(buf, stdout);
			}
		}
		if (ferror(f))
			perror(name);
		_fclose(f);
	}
	exit(0);
}

