#include <Runtime.h>
#include "futils.h"

int main(int argc, char **argv)
{
	atexit(RuntimeSentinel::ClientShutdown);
	RuntimeSentinel::ClientInitialize();
	int	mode = 0;
	char *cp = argv[1];
	while (isoctal(*cp))
		mode = mode * 8 + (*cp++ - '0');
	if (*cp)
	{
		fprintf(stderr, "Mode must be octal\n");
		exit(1);
	}
	//
	argc--;
	argv++;
	while (argc-- > 1)
	{
		if (chmod(argv[1], mode) < 0)
			perror(argv[1]);
		argv++;
	}
	exit(0);
}
