#include "futils.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <pwd.h>
#include <grp.h>
#include <utime.h>
#include <errno.h>

int main(int argc, char **argv)
{
	char *cp;
	int gid;
	struct group *grp;
	struct stat	statbuf;

	cp = argv[1];
	if (isdecimal(*cp)) {
		gid = 0;
		while (isdecimal(*cp))
			gid = gid * 10 + (*cp++ - '0');

		if (*cp) {
			fprintf(stderr, "Bad gid value\n");
			exit(1);
		}
	} else {
		grp = getgrnam(cp);
		if (grp == NULL) {
			fprintf(stderr, "Unknown group name\n");
			exit(1);
		}

		gid = grp->gr_gid;
	}

	argc--;
	argv++;

	while (argc-- > 1) {
		argv++;
		if ((stat(*argv, &statbuf) < 0) ||
			(chown(*argv, statbuf.st_uid, gid) < 0))
			perror(*argv);
	}
	exit(0);
}
