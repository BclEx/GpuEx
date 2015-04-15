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

int main(int argc, char	**argv)
{
	char		*cp;
	int uid;
	struct passwd *pwd;
	struct stat	statbuf;

	cp = argv[1];
	if (isdecimal(*cp)) {
		uid = 0;
		while (isdecimal(*cp))
			uid = uid * 10 + (*cp++ - '0');

		if (*cp) {
			fprintf(stderr, "Bad uid value\n");
			exit(1);
		}
	} else {
		pwd = getpwnam(cp);
		if (pwd == NULL) {
			fprintf(stderr, "Unknown user name\n");
			exit(1);
		}

		uid = pwd->pw_uid;
	}

	argc--;
	argv++;

	while (argc-- > 1) {
		argv++;
		if ((stat(*argv, &statbuf) < 0) ||
			(chown(*argv, uid, statbuf.st_gid) < 0))
				perror(*argv);
	}
	exit(0);
}
