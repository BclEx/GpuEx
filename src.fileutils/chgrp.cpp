#include <Runtime.h>
#include "futils.h"
//
#include <sys/stat.h>

int main(int argc, char **argv)
{
	atexit(RuntimeSentinel::ClientShutdown);
	RuntimeSentinel::ClientInitialize();
	char *cp = argv[1];
	int gid;
	struct group *grp;
	if (isdecimal(*cp))
	{
		gid = 0;
		while (isdecimal(*cp))
			gid = gid * 10 + (*cp++ - '0');
		if (*cp)
		{
			fprintf(stderr, "Bad gid value\n");
			exit(1);
		}
	}
	else
	{
		grp = getgrnam(cp);
		if (!grp)
		{
			fprintf(stderr, "Unknown group name\n");
			exit(1);
		}
		gid = grp->gr_gid;
	}
	//
	argc--;
	argv++;
	while (argc-- > 1)
	{
		argv++;
		struct stat	statbuf;
		if (stat(*argv, &statbuf) < 0 || chown(*argv, statbuf.st_uid, gid) < 0)
			perror(*argv);
	}
	exit(0);
}
