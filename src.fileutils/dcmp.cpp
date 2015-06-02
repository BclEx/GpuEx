#include <Runtime.h>
#include "futils.h"
#include <sys/stat.h>

int main(int argc, char	**argv)
{
	atexit(RuntimeSentinel::ClientShutdown);
	RuntimeSentinel::ClientInitialize();
	struct stat statbuf1;
	if (stat(argv[1], &statbuf1) < 0)
	{
		perror(argv[1]);
		exit(2);
	}
	struct stat statbuf2;
	if (stat(argv[2], &statbuf2) < 0)
	{
		perror(argv[2]);
		exit(2);
	}
	if (statbuf1.st_dev == statbuf2.st_dev && statbuf1.st_ino == statbuf2.st_ino)
	{
		printf("Files are links to each other\n");
		exit(0);
	}
	if (statbuf1.st_size != statbuf2.st_size)
	{
		printf("Files are different sizes\n");
		exit(1);
	}
	FILE *f1 = _fopen(argv[1], "r");
	if (!f1)
	{
		perror(argv[1]);
		exit(2);
	}
	FILE *f2 = _fopen(argv[2], "r");
	if (!f2)
	{
		perror(argv[2]);
		_fclose(f1);
		exit(2);
	}
	//
	long pos = 0;
	char buf1[512];
	char buf2[512];
	char *bp1;
	char *bp2;
	while (1)
	{
		size_t cc1 = _fread(buf1, 1, sizeof(buf1), f1);
		if (cc1 < 0)
		{
			perror(argv[1]);
			goto eof;
		}
		size_t cc2 = _fread(buf2, 1, sizeof(buf2), f2);
		if (cc2 < 0)
		{
			perror(argv[2]);
			goto differ;
		}
		if (cc1 == 0 && cc2 == 0)
		{
			printf("Files are identical\n");
			goto same;
		}
		if (cc1 < cc2)
		{
			printf("First file is shorter than second\n");
			goto differ;
		}
		if (cc1 > cc2)
		{
			printf("Second file is shorter than first\n");
			goto differ;
		}
		if (!memcmp(buf1, buf2, cc1))
		{
			pos += (long)cc1;
			continue;
		}
		//
		bp1 = buf1;
		bp2 = buf2;
		while (*bp1++ == *bp2++)
			pos++;
		printf("Files differ at byte position %ld\n", pos);
		goto differ;
	}
eof:
	_fclose(f1);
	_fclose(f2);
	exit(2);
same:
	_fclose(f1);
	_fclose(f2);
	exit(0);
differ:
	_fclose(f1);
	_fclose(f2);
	exit(1);
}
