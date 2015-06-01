#include <Runtime.h>
#include "futils.h"
#include <sys/stat.h>

int main(int argc, char	**argv)
{
	atexit(RuntimeSentinel::ClientShutdown);
	RuntimeSentinel::ClientInitialize();
	char *cp = argv[1];
	int uid;
	if (isdecimal(*cp))
	{
		uid = 0;
		while (isdecimal(*cp))
			uid = uid * 10 + (*cp++ - '0');
		if (*cp)
		{
			fprintf(stderr, "Bad uid value\n");
			exit(1);
		}
	}
	else
	{
		struct passwd *pwd = getpwnam(cp);
		if (!pwd)
		{
			fprintf(stderr, "Unknown user name\n");
			exit(1);
		}
		uid = pwd->pw_uid;
	}
	//
	argc--;
	argv++;
	while (argc-- > 1)
	{
		argv++;
		struct stat	statbuf;
		if ((stat(*argv, &statbuf) < 0) || (chown(*argv, uid, statbuf.st_gid) < 0))
			perror(*argv);
	}
	exit(0);
}
